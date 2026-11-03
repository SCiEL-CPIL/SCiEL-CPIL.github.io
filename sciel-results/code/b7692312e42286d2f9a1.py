import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    theta = np.radians(angle_deg)
    angles = theta + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _polygons_overlap(poly1, poly2, tol=1e-7):
    """SAT overlap test for two convex polygons."""
    for poly in (poly1, poly2):
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            nx, ny = -dy, dx
            proj1 = np.dot(poly1, [nx, ny])
            proj2 = np.dot(poly2, [nx, ny])
            if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                return False
    return True

def _min_outer_side(inner, ocx, ocy, oangle_deg):
    """Minimum outer hexagon side to contain all inner hexagons."""
    theta = np.radians(oangle_deg)
    normals = np.column_stack([
        np.cos(theta + (np.arange(6) + 0.5) * np.pi / 3.0),
        np.sin(theta + (np.arange(6) + 0.5) * np.pi / 3.0)
    ])
    max_s = 0.0
    for row in inner:
        ix, iy, ia = row
        verts = _hex_vertices(ix, iy, 1.0, ia)
        for v in verts:
            rel = v - np.array([ocx, ocy])
            dists = np.dot(normals, rel)
            req = np.max(np.maximum(dists, 0)) * 2.0 / np.sqrt(3.0)
            if req > max_s:
                max_s = req
    return max_s

def _cost(params, ocx, ocy, oangle):
    inner = []
    for i in range(N):
        inner.append([params[3*i], params[3*i+1], params[3*i+2]])
    side = _min_outer_side(inner, ocx, ocy, oangle)
    pen = 0.0
    for i in range(N):
        vi = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for j in range(i+1, N):
            vj = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if _polygons_overlap(vi, vj):
                pen += 50.0
    return side + pen

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 140.0
    sqrt3 = np.sqrt(3.0)
    
    # Initial: 3 rows of 4 hexagons in hex grid
    init = []
    for r in range(3):
        for c in range(4):
            x = (c - 1.5) * sqrt3 + (r % 2) * (sqrt3 / 2)
            y = r * 1.5
            init.append([x, y, 0.0])
    # Center
    xs = [p[0] for p in init]
    ys = [p[1] for p in init]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for i in range(N):
        init[i][0] -= cx
        init[i][1] -= cy
    
    oc = (0.0, 0.0)
    oa = 0.0
    
    x0 = np.array([v for row in init for v in row])
    
    def obj(p):
        return _cost(p, oc[0], oc[1], oa)
    
    best = x0.copy()
    best_c = obj(x0)
    
    res = minimize(obj, x0, method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
    if res.fun < best_c:
        best, best_c = res.x, res.fun
    
    # Extra restarts if time permits
    for s in range(5):
        if time.time() - t0 > deadline - 15:
            break
        rng = np.random.RandomState(1000 + s)
        xk = best + rng.normal(0, 0.3, len(best))
        rk = minimize(obj, xk, method='Nelder-Mead',
                      options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
        if rk.fun < best_c:
            best, best_c = rk.x, rk.fun
    
    inner = [[float(best[3*i]), float(best[3*i+1]), float(best[3*i+2])] for i in range(N)]
    side = _min_outer_side(inner, oc[0], oc[1], oa)
    return inner, oc, side, oa
# EVOLVE-BLOCK-END
