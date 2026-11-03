"""Fresh design: continuous soft-min optimization of inner hex positions + LP outer polish."""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 11
_DEADLINE = 150.0
_T0 = time.time()

# 6 outward normals for a regular hexagon of side s centered at c, rotated by a:
# face i is at distance s from center; outward normal n_i; constraint n_i . p <= s + n_i . c
def _normals(a):
    return np.array([[np.cos(a + k * np.pi / 3.0) for k in range(6)],
                     [np.sin(a + k * np.pi / 3.0) for k in range(6)]]).T  # 6x2

def _vertices(px, py, a):
    # 6 vertices of a unit regular hexagon centered at (px,py), rotated by a
    r = 1.0
    ang = a + np.arange(6) * np.pi / 3.0
    return np.array([px + r * np.cos(ang), py + r * np.sin(ang)]).T  # 6x2

def _min_dist(A, B):
    # minimum distance between two convex hexagons (6x2 each), or 0 if overlap
    # check vertex-edge distances both ways
    best = np.inf
    for i in range(6):
        a0, a1 = A[i], A[(i + 1) % 6]
        for j in range(6):
            b0, b1 = B[j], B[(j + 1) % 6]
            # point a0 to segment b0-b1
            ab = b1 - b0
            t = np.clip(np.dot(a0 - b0, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
            d = np.linalg.norm(a0 - (b0 + t * ab))
            if d < best:
                best = d
            # point b0 to segment a0-a1
            ba = a1 - a0
            t2 = np.clip(np.dot(b0 - a0, ba) / max(np.dot(ba, ba), 1e-12), 0, 1)
            d2 = np.linalg.norm(b0 - (a0 + t2 * ba))
            if d2 < best:
                best = d2
    if best < 1e-9:
        return 0.0
    return best

def _softmin(vals, t):
    # soft-min of vals via log-sum-exp with temperature t
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return 0.0
    m = vals.min()
    if m < -1e6:
        return m
    return m - t * np.log(np.sum(np.exp((vals - m) / t)))

def _objective(x, t=0.05):
    # x: 2*N positions
    pos = x.reshape(N, 2)
    # precompute vertices for all inner hexes (angle 0 for speed)
    verts = [_vertices(pos[i, 0], pos[i, 1], 0.0) for i in range(N)]
    # pairwise soft-min distances
    d = []
    for i in range(N):
        for j in range(i + 1, N):
            d.append(_min_dist(verts[i], verts[j]))
    # soft-min pairwise (want all >= 0, penalize overlaps)
    pair_pen = _softmin(np.array(d), t)
    # soft-min distance from each inner vertex to outer boundary (outer side 1, centered 0, angle 0)
    # outer: 6 faces, outward normals at angles k*pi/3, constraint n.p <= 1
    normals = _normals(0.0)
    bd = []
    for i in range(N):
        for v in verts[i]:
            # distance to each face: 1 - n.v (positive if inside)
            for n in normals:
                bd.append(1.0 - float(np.dot(n, v)))
    outer_pen = _softmin(np.array(bd), t)
    # want both large; objective = negative soft-mins (minimize)
    return -(pair_pen + outer_pen)

def _construct_positions(pos):
    inner = [[float(pos[i, 0]), float(pos[i, 1]), 0.0] for i in range(N)]
    return inner

def _lp_outer(inner_pos):
    # Find minimum side length s of a regular hexagon (center c, angle a) enclosing all inner hexes.
    # We do a coarse grid over angle and center, then LP for s.
    # For fixed (c, a), min s such that for each inner vertex v and each normal n_k:
    #   n_k . v <= s + n_k . c   =>  s >= max over (v,k) of (n_k . v - n_k . c)
    # So for fixed (c,a): s = max_{v,k} (n_k . (v - c))
    # We minimize over (c, a) on a grid.
    best = (np.inf, 0.0, 0.0, 0.0)
    pts = []
    for (px, py, _) in inner_pos:
        for v in _vertices(px, py, 0.0):
            pts.append(v)
    pts = np.array(pts)
    # grid over angle (0..pi/3) and center
    ang_grid = np.linspace(0, np.pi / 3.0, 13)
    cx_range = np.linspace(-2, 2, 9)
    cy_range = np.linspace(-2, 2, 9)
    for a in ang_grid:
        normals = _normals(a)
        for cx in cx_range:
            for cy in cy_range:
                c = np.array([cx, cy])
                # s = max over v,k of n_k . (v - c)
                s = 0.0
                for v in pts:
                    for n in normals:
                        val = float(np.dot(n, v - c))
                        if val > s:
                            s = val
                if s < best[0]:
                    best = (s, cx, cy, float(a))
    return best

def construct():
    global _T0
    elapsed = time.time() - _T0
    if elapsed > _DEADLINE - 5:
        # fallback: trivial
        inner = _construct_positions(np.zeros((N, 2)))
        return inner, (0.0, 0.0), 10.0, 0.0

    # Initial: hexagonal lattice packing (dense)
    # hex lattice spacing for touching unit hexes: distance between centers = 2*sqrt(3)/... 
    # For flat-top hexes side 1, width = 2, height = sqrt(3). Touching centers: 
    # horizontal spacing 2*sqrt(3) (for pointy) ... use known dense: centers on triangular lattice
    # side of hex = 1, inradius = sqrt(3)/2, circumradius = 1.
    # Nearest-neighbor center distance for non-overlap = 2 * (sqrt(3)/2) * 2? 
    # Actually two unit hexes touching: center distance = 2 * inradius * 2? Let's use 2*sqrt(3) approx.
    # Use triangular lattice with spacing 2*sqrt(3) ~ 3.464
    sp = 2 * np.sqrt(3)
    rows = []
    r = 2
    for i in range(N):
        row = i // 3
        col = i % 3
        if row % 2 == 1:
            x = (col - 1) * sp + sp / 2
        else:
            x = (col - 1) * sp
        y = row * sp * np.sqrt(3) / 2
        rows.append([x, y])
    x0 = np.array(rows).reshape(-1)

    best_obj = np.inf
    best_x = x0.copy()
    t0 = time.time()
    # multi-start L-BFGS-B
    rng = np.random.default_rng(0)
    n_starts = 1
    # first run from lattice
    for start in range(n_starts):
        if time.time() - _T0 > _DEADLINE - 30:
            break
        if start == 0:
            x_init = x0.copy()
        else:
            x_init = x0 + rng.normal(0, 0.5, size=x0.shape)
        res = minimize(_objective, x_init, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x
        if time.time() - _T0 > _DEADLINE - 40:
            break

    pos = best_x.reshape(N, 2)
    inner = _construct_positions(pos)

    # LP outer polish
    if time.time() - _T0 < _DEADLINE - 10:
        s, cx, cy, a = _lp_outer(inner)
        if not np.isfinite(s):
            s, cx, cy, a = 10.0, 0.0, 0.0, 0.0
    else:
        s, cx, cy, a = 10.0, 0.0, 0.0, 0.0

    return inner, (float(cx), float(cy)), float(s), float(a)

# Safety: ensure construct is callable
if __name__ == "__main__":
    r = construct()
    print("done", len(r[0]))
