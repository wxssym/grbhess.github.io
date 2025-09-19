#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd

from math import pi
from bokeh.palettes import Category20c, Category10
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, HoverTool

# ---------- config ----------
CSV_PATH = os.environ.get("OBS_CSV", "obstable.csv")         # repo root
DONUT_HTML = os.environ.get("DONUT_HTML", "donut.html")         # repo root
MAP_HTML   = os.environ.get("MAP_HTML",   "map.html")           # repo root

DB_HOST = os.environ.get("DB_HOST")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_NAME = os.environ.get("DB_NAME", "ANALYSIS")

# ---------- helpers ----------
def normalize_cat(s: str) -> str:
    s = str(s).strip()
    s = s.replace("\u200b", "")
    s = s.replace("–", "-").replace("—", "-").replace("−", "-").replace("/", "-")
    s = " ".join(s.split())
    return s.upper()

def choose_palette(n):
    if n <= 0: return []
    if n == 1: return ["#1f77b4"]
    if n == 2: return ["#1f77b4", "#ff7f0e"]
    if n <= 10: return Category10[max(3, n)][:n]
    if n <= 20: return Category20c[n]
    base = Category20c[20]
    reps = (n // 20) + 1
    return (base * reps)[:n]

def prepare_donut_data(data_dict):
    data = pd.Series(data_dict).reset_index(name='value').rename(columns={'index': 'category'})
    tot = data['value'].sum()
    data['angle'] = data['value'] / (tot if tot else 1) * 2 * pi
    data['color'] = choose_palette(len(data))
    return data

def donut_plot(data, title):
    from bokeh.transform import cumsum
    p = figure(height=450, width=450, title=title, toolbar_location=None,
               tools="hover", tooltips="@category: @value", x_range=(-0.6, 1.2))
    p.wedge(x=0, y=1, radius=0.55,
            start_angle=cumsum('angle', include_zero=True),
            end_angle=cumsum('angle'),
            line_color="white", fill_color='color',
            legend_field='category', source=data)
    p.annulus(x=0, y=1, inner_radius=0.30, outer_radius=0.55, color='white')
    p.axis.visible = False
    p.grid.grid_line_color = None
    return p

# Sky map projection helpers (Mollweide with 180→0→−180)
def radians(deg):
    return np.deg2rad(deg)

def mollweide_xy(ra_deg, dec_deg):
    # Shift RA to 180,0,-180 convention: lon_deg in [-180,180]
    lon = ((180.0 - ra_deg + 180.0) % 360.0) - 180.0
    lat = dec_deg
    # Convert to radians
    lam = radians(lon)
    phi = radians(lat)
    # Solve for theta using Newton iteration
    theta = phi.copy()
    for _ in range(10):
        theta -= (2*theta + np.sin(2*theta) - pi*np.sin(phi)) / (2 + 2*np.cos(2*theta))
    # Mollweide
    x = (2*np.sqrt(2)/pi) * lam * np.cos(theta)
    y = np.sqrt(2) * np.sin(theta)
    return x, y

def make_sky_map(df, title="GRB Sky Map (Mollweide)"):
    # RA/Dec in degrees from CSV columns already merged into df
    ra = pd.to_numeric(df["GRB RA (J2000)"], errors="coerce")
    dec = pd.to_numeric(df["GRB Dec (J2000)"], errors="coerce")
    m = ra.notna() & dec.notna()
    ra = ra[m].values
    dec = dec[m].values
    x, y = mollweide_xy(ra, dec)
    # Bokeh figure
    p = figure(height=700, width=1200, title=title, toolbar_location="above",
               x_range=(-2.2, 2.2), y_range=(-1.2, 1.2),
               tools="pan,wheel_zoom,reset,save,hover")
    p.grid.grid_line_alpha = 0.15
    p.xaxis.visible = False
    p.yaxis.visible = False
    # Optional graticule (lon every 60°, lat every 30°)
    for lat in [-60, -30, 0, 30, 60]:
        lon_line = np.linspace(-180, 180, 361)
        lat_line = np.full_like(lon_line, lat, dtype=float)
        gx, gy = mollweide_xy(180 - lon_line, lat_line)  # invert back to RA for the helper signature
        p.line(gx, gy, line_color="#dddddd", line_alpha=0.6)
    for lon in [-180, -120, -60, 0, 60, 120, 180]:
        lat_line = np.linspace(-90, 90, 361)
        lon_line = np.full_like(lat_line, lon, dtype=float)
        gx, gy = mollweide_xy(180 - lon_line, lat_line)
        p.line(gx, gy, line_color="#dddddd", line_alpha=0.6)

    # Points
    src = ColumnDataSource(dict(
        x=x, y=y,
        ra=ra, dec=dec,
        target=df.loc[m, "Target"].astype(str).values,
        trig=df.loc[m, "Triggering instrument"].astype(str).values,
        react=df.loc[m, "Reaction"].astype(str).values,
    ))
    p.circle("x", "y", size=7, alpha=0.9, color="#1f77b4", line_color="white", source=src)
    p.add_tools(HoverTool(tooltips=[("Target", "@target"),
                                    ("RA", "@ra{0.000}"),
                                    ("Dec", "@dec{0.000}"),
                                    ("Trigger", "@trig"),
                                    ("Reaction", "@react")]))
    return p

# ---------- load CSV from repo root ----------
cols = ["GRB ID","Triggering instrument","Alert time (T0)","GRB RA (J2000)","GRB Dec (J2000)",
        "H.E.S.S. window start","H.E.S.S. window end","Obs mode","Reaction","Contact"]
repo_grbs = pd.read_csv(CSV_PATH, header=None, names=cols)

# ---------- fetch DB data using secrets ----------
import MySQLdb
conn = MySQLdb.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, passwd=DB_PASS, db=DB_NAME)
cur = conn.cursor()
tables = ["RunQualitySet","RunQuality202106Set","RunQuality202402Set"]
rows = []
for t in tables:
    q = f"""
    SELECT RunNumber, Target, NominalTargetRa, NominalTargetDec,
           NominalWoobleOffsetRa, NominalWoobleOffsetDec
    FROM {t}
    WHERE NominalTargetRa IS NOT NULL AND NominalTargetDec IS NOT NULL
      AND (Target LIKE '%ToO%GRB%' OR Target LIKE 'GRB%')
    """
    cur.execute(q)
    rows.extend(cur.fetchall())
cur.close(); conn.close()

grb_summary = pd.DataFrame(rows, columns=[
    "RunNumber","Target","TargetRA","TargetDec","WoobleRA","WoobleDec"
])

# ---------- match to repo GRBs by sanitized IDs and merge metadata ----------
def sanitize_grb_id(s):
    m = re.search(r"(\d{6}[A-Za-z]?)", str(s))
    return m.group(1) if m else ""

repo_grbs["sanitized_id"] = repo_grbs["GRB ID"].map(sanitize_grb_id)
grb_summary["sanitized_id"] = grb_summary["Target"].map(sanitize_grb_id)

grb_summary = grb_summary.merge(
    repo_grbs[["sanitized_id","Triggering instrument","Reaction","Obs mode",
               "GRB RA (J2000)","GRB Dec (J2000)"]],
    on="sanitized_id", how="inner"
)

# ---------- build donuts ----------
triggering_counts = grb_summary["Triggering instrument"].map(normalize_cat).value_counts().to_dict()
reaction_counts   = grb_summary["Reaction"].map(normalize_cat).fillna("UNKNOWN").value_counts().to_dict()

triggering_data = prepare_donut_data(triggering_counts)
reaction_data   = prepare_donut_data(reaction_counts)

p1 = donut_plot(triggering_data, "Triggering Instruments")
p2 = donut_plot(reaction_data, "Reaction")

output_file(DONUT_HTML)
save(row(p1, p2))

# ---------- build sky map with same projection ----------
sky = make_sky_map(grb_summary, title="GRB Sky Map (Mollweide 180→0→−180)")
output_file(MAP_HTML)
save(sky)
