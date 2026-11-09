import time
import math
import numpy as np

N = 11
DEADLINE = 140.0
_START = time.time()
_EPS = 1e-9

# --- geometry helpers -------------------------------------------------------
def _hex_verts(cx, cy, ang):
    a = math.radians(ang)
    c, s = math.cos(a), math.sin(a)
    out = []
    for i in range(6):
        t = math.radians(60.0 * i)
        x = math.cos(t)
        y = math.sin(t)
        out.append((cx + c * x - s * y, cy + s * x + c * y))
    return out

def _seg_seg(p1, p2, p3, p4):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return False

def _pts_intersect(a, b):
    for i in range(len(a)):
        for j in range(len(b)):
            if _seg_seg(a[i], a[(i + 1) % len(a)], b[j], b[(j + 1) % len(b)]):
                return True
    return False

def _in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside

def _colliding(a, b):
    if _pts_intersect(a, b):
        return True
    if _in_poly(a[0], b):
        return True
    if _in_poly(b[0], a):
        return True
    return False

# --- smooth objective -------------------------------------------------------
def _smooth_loss(xs, centers, angs, outer_cx, outer_cy, outer_s, outer_a):
    m = len(centers)
    loss = 0.0
    # collisions between inner hexagons
    for i in range(m):
        for j in range(i + 1, m):
            d = math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            loss += 0.5 * math.exp(2.0 * (d - 2.0))
    # inner vs outer boundary (inner hex must stay inside outer hex)
    for i in range(m):
        for k in range(6):
            t = math.radians(60.0 * k)
            ox = outer_cx + outer_s * math.cos(outer_a + t)
            oy = outer_cy + outer_s * math.sin(outer_a + t)
            d = math.hypot(centers[i][0] - ox, centers[i][1] - oy)
            loss += 0.5 * math.exp(1.2 * (d - outer_s))
    return loss

# --- optimizer --------------------------------------------------------------
def _optimize(centers, angs, outer_cx, outer_cy, outer_s, outer_a, budget):
    m = len(centers)
    nvar = 2 * m + 1  # centers (2m) + outer side
    x0 = np.array([c[0] for c in centers] + [c[1] for c in centers] + [outer_s], dtype=float)
    bounds = [(-60.0, 60.0)] * (2 * m) + [(1.0, 60.0)]

    def fun(x):
        c = x[:2 * m].reshape(m, 2)
        s = x[2 * m]
        return _smooth_loss(x, c, angs, outer_cx, outer_cy, s, outer_a)

    from scipy.optimize import minimize
    res = minimize(fun, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 250, 'ftol': 1e-9, 'gtol': 1e-6, 'maxfun': 4000})
    xf = res.x
    c = xf[:2 * m].reshape(m, 2)
    s = float(xf[2 * m])
    return c, s

# --- validity check ---------------------------------------------------------
def _valid(inner, center, side, angle):
    outer = _hex_verts(center[0], center[1], angle)
    polys = [_hex_verts(p[0], p[1], p[2]) for p in inner]
    for p in polys:
        if not _in_poly(p[0], outer):
            return False
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if _colliding(polys[i], polys[j]):
                return False
    return True

# --- initial topologies -----------------------------------------------------
def _topo_flat(cx, cy, s):
    centers, angs = [], []
    for k in range(-N // 2, N - N // 2):
        centers.append((cx + 2.0 * s * k, cy))
        angs.append(0.0)
    return centers, angs

def _topo_rot(cx, cy, s, ang):
    centers, angs = [], []
    for k in range(-N // 2, N - N // 2):
        centers.append((cx + 2.0 * s * k, cy))
        angs.append(ang)
    return centers, angs

def _topo_row(cx, cy, s, ang):
    centers, angs = [], []
    for k in range(-N // 2, N - N // 2):
        centers.append((cx + 2.0 * s * k, cy))
        angs.append(ang)
    return centers, angs

def _topo_cluster(cx, cy, s, ang):
    centers, angs = [], []
    # hexagonal cluster of N cells around origin, spacing 2s
    d = 2.0 * s
    rings = []
    rings.append((0, 0))
    for q in range(-2, 3):
        for r in range(-2, 3):
            if q == 0 and r == 0:
                continue
            if abs(q + r) <= 2 and abs(q) <= 2 and abs(r) <= 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                rings.append((x, y))
    rings = rings[:N]
    for (x, y) in rings:
        centers.append((cx + x, cy + y))
        angs.append(ang)
    return centers, angs

def _topo_stagger(cx, cy, s, ang):
    centers, angs = [], []
    d = 2.0 * s
    # two staggered rows
    per = (N + 1) // 2
    for row in range(2):
        yoff = (row - 0.5) * d * math.sqrt(3) / 2.0
        for k in range(per):
            xoff = d * (k - (per - 1) / 2.0) + (row % 2) * d / 2.0
            centers.append((cx + xoff, cy + yoff))
            angs.append(ang)
    while len(centers) < N:
        centers.append((cx, cy))
        angs.append(ang)
    return centers[:N], angs[:N]

def _starts():
    starts = []
    starts.append(_topo_flat(0.0, 0.0, 2.0))
    starts.append(_topo_flat(0.0, 0.0, 2.0))
    for a in [15.0, 30.0, 45.0]:
        starts.append(_topo_rot(0.0, 0.0, 2.0, a))
    starts.append(_topo_cluster(0.0, 0.0, 2.0, 0.0))
    starts.append(_topo_cluster(0.0, 0.0, 2.0, 30.0))
    starts.append(_topo_stagger(0.0, 0.0, 2.0, 0.0))
    starts.append(_topo_stagger(0.0, 0.0, 2.0, 15.0))
    return starts

def construct():
    best = None
    best_side = float('inf')
    outer_cx, outer_cy, outer_a = 0.0, 0.0, 0.0
    starts = _starts()
    for centers, angs in starts:
        if time.time() > DEADLINE:
            break
        try:
            c, s = _optimize(centers, angs, outer_cx, outer_cy, 6.0, outer_a, 0.0)
        except Exception:
            continue
        inner = [[float(c[i][0]), float(c[i][1]), float(angs[i])] for i in range(len(angs))]
        if _valid(inner, (outer_cx, outer_cy), s, outer_a) and s < best_side:
            best_side = s
            best = (inner, (outer_cx, outer_cy), s, outer_a)
        # refine with the found side as a tighter start
        if time.time() > DEADLINE:
            break
        try:
            c2, s2 = _optimize(c, angs, outer_cx, outer_cy, s, outer_a, 0.0)
            inner2 = [[float(c2[i][0]), float(c2[i][1]), float(angs[i])] for i in range(len(angs))]
            if _valid(inner2, (outer_cx, outer_cy), s2, outer_a) and s2 < best_side:
                best_side = s2
                best = (inner2, (outer_cx, outer_cy), s2, outer_a)
        except Exception:
            pass

    if best is None:
        # safe fallback: trivially valid spread-out layout
        inner = [[float(2.0 * k), 0.0, 0.0] for k in range(-N // 2, N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    return best
