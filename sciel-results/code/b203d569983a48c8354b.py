# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(np.arange(6) * 60.0)
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])

def _edge_normals(V):
    n = V.shape[0]
    N = np.empty((n, 2))
    for i in range(n):
        p1 = V[i]
        p2 = V[(i + 1) % n]
        d = p2 - p1
        L = np.hypot(d[0], d[1])
        N[i] = np.array([-d[1], d[0]]) / L
    return N

def _separation(V1, V2):
    N1 = _edge_normals(V1)
    N2 = _edge_normals(V2)
    N = np.vstack([N1, N2])
    best = -np.inf
    for n in N:
        p1 = V1 @ n
        p2 = V2 @ n
        lo1, hi1 = p1.min(), p1.max()
        lo2, hi2 = p2.min(), p2.max()
        gap = max(lo2 - hi1, lo1 - hi2)
        if gap > best:
            best = gap
    return best

def _validate(x):
    ocx, ocy, oang, s = x[0], x[1], x[2], x[3]
    inner = x[4:].reshape(11, 3)
    if not np.all(np.isfinite(x)):
        return False
    if s < np.sqrt(11.0):
        return False
    m = 11
    V = [_hex_vertices(c[0], c[1], 1.0, c[2]) for c in inner]
    for i in range(m):
        for j in range(i + 1, m):
            if _separation(V[i], V[j]) < 1e-6:
                return False
    O = _hex_vertices(ocx, ocy, s, oang)
    for Vi in V:
        for i in range(6):
            p1 = O[i]
            p2 = O[(i + 1) % 6]
            d = p2 - p1
            for p in Vi:
                cr = d[0] * (p[1] - p1[1]) - d[1] * (p[0] - p1[0])
                if cr < -1e-6:
                    return False
    return True

def _min_side_for_config(x):
    ocx, ocy, oang = x[0], x[1], x[2]
    inner = x[4:].reshape(11, 3)
    V = [_hex_vertices(c[0], c[1], 1.0, c[2]) for c in inner]
    lo, hi = np.sqrt(11.0), 15.0
    # ensure hi is valid (large enough)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        O = _hex_vertices(ocx, ocy, mid, oang)
        ok = True
        for Vi in V:
            for i in range(6):
                p1 = O[i]
                p2 = O[(i + 1) % 6]
                d = p2 - p1
                for p in Vi:
                    cr = d[0] * (p[1] - p1[1]) - d[1] * (p[0] - p1[0])
                    if cr < -1e-6:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            hi = mid
        else:
            lo = mid
    return hi

def _objective(x):
    ocx, ocy, oang, s = x[0], x[1], x[2], x[3]
    inner = x[4:].reshape(11, 3)
    if s < np.sqrt(11.0) - 1e-3:
        return 1e6
    m = 11
    V = [_hex_vertices(c[0], c[1], 1.0, c[2]) for c in inner]
    sep = np.inf
    for i in range(m):
        for j in range(i + 1, m):
            g = _separation(V[i], V[j])
            if g < sep:
                sep = g
    sep_pen = 0.0
    if sep < 1e-6:
        sep_pen = (1e-6 - sep) * 1e4
    O = _hex_vertices(ocx, ocy, s, oang)
    cont_pen = 0.0
    for Vi in V:
        for i in range(6):
            p1 = O[i]
            p2 = O[(i + 1) % 6]
            d = p2 - p1
            for p in Vi:
                cr = d[0] * (p[1] - p1[1]) - d[1] * (p[0] - p1[0])
                if cr < -1e-6:
                    cont_pen += (-1e-6 - cr) * 1e4
    return 1.0 / s + sep_pen + cont_pen

def _build_hexagon_packing(outer_side, outer_angle_deg=0.0):
    a1 = np.array([2.0, 0.0])
    a2 = np.array([1.0, np.sqrt(3.0)])
    lattice = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1),
        (2, 0), (2, 1), (-2, 0), (-2, -1), (0, 2),
    ]
    pts = np.array([i * a1 + j * a2 for (i, j) in lattice])
    c = pts.mean(axis=0)
    pts = pts - c
    inner = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(11)])
    outer = np.array([0.0, 0.0, float(outer_angle_deg)])
    return inner, outer, float(outer_side)

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 600.0 - 30.0

    inner0, outer0, s0 = _build_hexagon_packing(6.0)
    x0 = np.concatenate([[0.0, 0.0, 0.0, 6.0], inner0.ravel()])

    best_x = x0.copy()
    best_val = 1.0 / best_x[3]

    # Optimize
    try:
        res = minimize(_objective, x0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-12})
        if res.x is not None and np.all(np.isfinite(res.x)) and _validate(res.x):
            best_x = res.x.copy()
    except Exception:
        pass

    if time.time() > deadline:
        pass

    # Shrink side
    new_s = _min_side_for_config(best_x)
    best_x[3] = new_s
    best_val = 1.0 / new_s

    # Restarts
    for restart in range(4):
        if time.time() > deadline:
            break
        rng = np.random.default_rng(restart + 7)
        x_try = best_x.copy()
        x_try[4:] += rng.normal(0, 0.08, size=x_try[4:].shape)
        x_try[3] = best_x[3] + 0.3
        try:
            res = minimize(_objective, x_try, method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-12})
            if res.x is not None and np.all(np.isfinite(res.x)) and _validate(res.x):
                ns = _min_side_for_config(res.x)
                if 1.0 / ns > best_val:
                    best_x = res.x.copy()
                    best_x[3] = ns
                    best_val = 1.0 / ns
        except Exception:
            pass

    ocx, ocy, oang, s = best_x[0], best_x[1], best_x[2], best_x[3]
    inner = best_x[4:].reshape(11, 3)
    outer = np.array([ocx, ocy, oang])
    return inner, outer, float(s)

# EVOLVE-BLOCK-END
