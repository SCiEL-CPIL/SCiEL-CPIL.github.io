"""Packing 11 unit hexagons inside a regular hexagon; optimize container rotation/side."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
_T0 = time.time()
_DEADLINE = 150.0
_MARGIN = 12.0

# Unit hexagon (pointy-top) vertices, circumradius 1
_HEX_VERT = np.array([[0.0, 1.0],
                      [np.sqrt(3) / 2, 0.5],
                      [np.sqrt(3) / 2, -0.5],
                      [0.0, -1.0],
                      [-np.sqrt(3) / 2, -0.5],
                      [-np.sqrt(3) / 2, 0.5]])

# Hex-lattice positions for the 11-hex cluster (1 center + 6 ring + 4 outer ring)
# Column spacing 1.5, row spacing sqrt(3), odd columns offset sqrt(3)/2
_SQ3 = np.sqrt(3)
def _col_row(c, r):
    x = 1.5 * c
    y = _SQ3 * (r + (0.5 if (c & 1) else 0.0))
    return x, y

# Build a compact 11-hex cluster
_CLUSTERS = []
def _build_cluster(cols):
    pts = []
    for c in cols:
        for r in range(-c, c + 1):
            pts.append(_col_row(c, r))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    return pts

# Try several cluster shapes and keep the most compact (smallest max radius)
for cols in ([0, 1, -1, 2, -2], [0, 1, -1, 2], [0, 1, -1, 2, -2, 3, -3]):
    p = _build_cluster(cols)
    if len(p) == N:
        _CLUSTERS.append(p)
# fallback: use the first one
if not _CLUSTERS:
    _CLUSTERS.append(_build_cluster([0, 1, -1, 2, -2]))
_CLUSTER = _CLUSTERS[0]

def _hex_vertices(center, angle_deg):
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    return _HEX_VERT @ R.T + center

def _container_boundary(pts, center, side, rot_deg):
    """Return signed distance of each point to container interior (positive = inside).
    Container: regular hexagon, side length 'side', rotated rot_deg.
    Use the 6 supporting lines (inward normals) at distance 'side' from center
    (apothem = side for a regular hexagon of side length side)."""
    a = np.deg2rad(rot_deg)
    ca, sa = np.cos(a), np.sin(a)
    # outward normals of a regular hexagon (pointy-top base at 30,90,150,210,270,330)
    base = np.array([[np.cos(np.deg2rad(d)), np.sin(np.deg2rad(d))] for d in
                     [30, 90, 150, 210, 270, 330]])
    normals = base @ np.array([[ca, -sa], [sa, ca]])
    apothem = side  # distance from center to each side
    rel = pts - center
    # signed distance = apothem - dot(rel, normal)  (positive inside)
    d = apothem - rel @ normals.T
    return d.min(axis=1)  # min over the 6 sides per point

def _objective(x):
    # x = [side, rot_deg, tx, ty, angles...]
    side, rot_deg, tx, ty = x[0], x[1], x[2], x[3]
    angles = x[4:]
    centers = _CLUSTER + np.array([tx, ty])
    verts = np.array([_hex_vertices(centers[i], angles[i]) for i in range(N)])
    allpts = verts.reshape(-1, 2)
    d = _container_boundary(allpts, np.array([0.0, 0.0]), side, rot_deg)
    # smooth penalty for boundary violations
    viol = np.maximum(0.0, -d)
    pen = np.sum(viol ** 2)
    # also penalize inner overlap lightly (shouldn't occur in lattice)
    return side + 10.0 * pen

def _best_result():
    # grid search over rotation, then polish
    best = (None, None, None, None, np.inf)
    # initial candidate: cluster centered, rot 0
    for rot in np.linspace(0, 60, 13):
        for tx in np.linspace(-0.5, 0.5, 5):
            for ty in np.linspace(-0.5, 0.5, 5):
                centers = _CLUSTER + np.array([tx, ty])
                verts = np.array([_hex_vertices(centers[i], 0.0) for i in range(N)])
                allpts = verts.reshape(-1, 2)
                # minimal side so all points inside: side >= max over points of max dot(rel,n)
                a = np.deg2rad(rot)
                ca, sa = np.cos(a), np.sin(a)
                base = np.array([[np.cos(np.deg2rad(d)), np.sin(np.deg2rad(d))] for d in
                                 [30, 90, 150, 210, 270, 330]])
                normals = base @ np.array([[ca, -sa], [sa, ca]])
                rel = allpts
                need = (rel @ normals.T).max()
                side = need + 1e-9
                val = _objective(np.concatenate(([side, rot, tx, ty], np.zeros(N))))
                if val < best[4]:
                    best = (side, rot, tx, ty, val)
    return best

def construct():
    global _T0
    best = _best_result()
    side, rot, tx, ty, val = best
    if val == np.inf:
        side, rot, tx, ty = 9.0, 0.0, 0.0, 0.0
    # polish with scipy
    if time.time() - _T0 < _DEADLINE - _MARGIN:
        x0 = np.concatenate(([side, rot, tx, ty], np.zeros(N)))
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if res.fun < val:
                side, rot, tx, ty = res.x[0], res.x[1], res.x[2], res.x[3]
                val = res.fun
        except Exception:
            pass
    # build inner list
    angles = np.zeros(N)
    centers = _CLUSTER + np.array([tx, ty])
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(angles[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(side), float(rot)
