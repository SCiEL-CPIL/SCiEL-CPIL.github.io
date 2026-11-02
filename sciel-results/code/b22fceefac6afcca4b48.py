import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 12

# Unit regular hexagon (pointy-top) vertices: radius 1, angles 30,90,150,210,270,330 deg
_UNIT_V = np.array([
    [np.cos(np.radians(30)),  np.sin(np.radians(30))],
    [np.cos(np.radians(90)),  np.sin(np.radians(90))],
    [np.cos(np.radians(150)), np.sin(np.radians(150))],
    [np.cos(np.radians(210)), np.sin(np.radians(210))],
    [np.cos(np.radians(270)), np.sin(np.radians(270))],
    [np.cos(np.radians(330)), np.sin(np.radians(330))],
])
# Six outward normals (pointy-top hexagon), angles 0,60,...,300 deg
_NORM = np.array([
    [np.cos(np.radians(0)),   np.sin(np.radians(0))],
    [np.cos(np.radians(60)),  np.sin(np.radians(60))],
    [np.cos(np.radians(120)), np.sin(np.radians(120))],
    [np.cos(np.radians(210-60)), np.sin(np.radians(210-60))],  # 180
    [np.cos(np.radians(240)), np.sin(np.radians(240))],
    [np.cos(np.radians(300)), np.sin(np.radians(300))],
])
_NORM = np.array([
    [np.cos(np.radians(a)), np.sin(np.radians(a))] for a in (0, 60, 120, 180, 240, 300)
])

def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])

def _hex_verts(cx, cy, ang):
    R = _rot(ang)
    return (cx + cy) * 0 + np.tile(np.array([cx, cy]), (6, 1)) + _UNIT_V @ R.T

def _extreme_points(inner):
    # inner: list of (x,y,ang)
    pts = []
    for (x, y, ang) in inner:
        pts.append(_hex_verts(x, y, ang))
    return np.vstack(pts) if pts else np.zeros((0, 2))

def _outer_side(inner):
    """Tightest containing regular hexagon (pointy-top, axis-aligned) side length."""
    pts = _extreme_points(inner)
    if pts.shape[0] == 0:
        return 0.0
    # For each of 6 normals, find max projection; side = max over faces of (p + 1/sqrt3)
    # A regular hexagon with center c and side s contains all points iff for each normal n_i:
    #   max_j (n_i . (p_j - c)) <= s / sqrt(3)
    # For axis-aligned (c=0) minimal s: s = max_i ( max_j n_i.p_j + min_j n_i.p_j ) / sqrt(3)
    # Actually width along normal n_i = max n_i.p - min n_i.p; for hexagon the support in
    # direction n_i is s/sqrt(3). So need s >= sqrt(3) * (max - min) for each i.
    # But that's width-based; the true minimal containing hexagon side is max over i of
    # (max_j n_i.p_j - min_j n_i.p_j) * sqrt(3)  only if centered optimally. With free
    # center, width formulation holds. So:
    sides = []
    for n in _NORM:
        proj = pts @ n
        sides.append((proj.max() - proj.min()) * np.sqrt(3))
    return float(max(sides))

def _constructive_side(inner):
    return _outer_side(inner)

def _make_obj(inner0, time_budget):
    def obj(z):
        inner = []
        for k in range(N):
            x, y, a = z[3*k], z[3*k+1], z[3*k+2]
            inner.append((x, y, a))
        return _constructive_side(inner)
    return obj

def _init_grid(ring):
    """Generate initial placements on a hexagonal-ish lattice scaled by ring."""
    inners = []
    for k in range(N):
        # spread around a circle of radius ring
        ang = 2 * np.pi * k / N
        x = ring * np.cos(ang)
        y = ring * np.sin(ang)
        inners.append([x, y, 0.0])
    return inners

def _init_line():
    inners = []
    for k in range(N):
        inners.append([2.0 * k, 0.0, 0.0])
    return inners

def _init_random(seed, spread):
    rng = np.random.default_rng(seed)
    inners = []
    for k in range(N):
        x = rng.uniform(-spread, spread)
        y = rng.uniform(-spread, spread)
        a = rng.uniform(0, np.pi)
        inners.append([x, y, a])
    return inners

def _optimize(inners, time_budget, maxiter=400):
    z0 = np.array([v for inn in inners for v in inn])
    bounds = [(-50, 50), (-50, 50), (-np.pi, np.pi)] * N
    obj = _make_obj(inners, time_budget)
    res = minimize(obj, z0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': maxiter, 'maxfun': 2000, 'maxls': 40})
    return res

def _to_inner(res):
    inner = []
    z = res.x
    for k in range(N):
        inner.append([float(z[3*k]), float(z[3*k+1]), float(z[3*k+2])])
    return inner

def construct():
    t_start = time.time()
    DEADLINE = 120.0
    SAFETY = 10.0

    best_inner = _init_line()
    best_side = _constructive_side(best_inner)

    # list of initial guesses
    starts = []
    starts.append(('line', _init_line()))
    for ring in (1.5, 2.0, 2.5, 3.0, 3.5):
        starts.append((f'ring{ring}', _init_grid(ring)))
    for s in range(8):
        starts.append((f'rand{s}', _init_random(1000 + s, 4.0)))

    for name, inners in starts:
        if time.time() - t_start > DEADLINE - SAFETY:
            break
        try:
            res = _optimize(inners, DEADLINE - SAFETY)
            if res.x is not None:
                inner = _to_inner(res)
                side = _constructive_side(inner)
                if side < best_side - 1e-9:
                    best_side = side
                    best_inner = inner
        except Exception:
            pass

    # Final: compute outer center as centroid of extreme points for cleanliness
    pts = _extreme_points(best_inner)
    if pts.shape[0] > 0:
        center = (float(pts[:,0].mean()), float(pts[:,1].mean()))
    else:
        center = (0.0, 0.0)
    side = _constructive_side(best_inner)

    inner_out = [[float(x), float(y), float(np.degrees(a) % 360.0)] for (x, y, a) in best_inner]
    return inner_out, center, float(side), 0.0
