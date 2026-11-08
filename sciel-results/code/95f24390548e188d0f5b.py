"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 150.0
EPS = 1e-9


def _hex_verts(cx, cy, side, ang_rad):
    """Vertices of a regular hexagon (flat-top at angle 0)."""
    a = np.deg2rad(ang_rad)
    th = a + np.arange(6) * np.pi / 3.0
    r = side
    return np.column_stack((cx + r * np.cos(th), cy + r * np.sin(th)))


def _poly_overlap_depth(A, B):
    """Max penetration depth of two convex polygons (0 if disjoint)."""
    m, n = A.shape[0], B.shape[0]
    best = 0.0
    for k in range(m):
        ax1, ay1 = A[k]
        ax2, ay2 = A[(k + 1) % m]
        ex, ey = ax2 - ax1, ay2 - ay1
        elen = np.hypot(ex, ey)
        if elen < 1e-12:
            continue
        nx, ny = ey / elen, -ex / elen
        d1 = np.dot(A - A[k], [nx, ny])
        d2 = np.dot(B - A[k], [nx, ny])
        mn1, mx1 = d1.min(), d1.max()
        mn2, mx2 = d2.min(), d2.max()
        if mx1 < mn2 + EPS or mx2 < mn1 + EPS:
            return 0.0
        best = max(best, min(mx1, mx2) - max(mn1, mn2))
    for k in range(n):
        ax1, ay1 = B[k]
        ax2, ay2 = B[(k + 1) % n]
        ex, ey = ax2 - ax1, ay2 - ay1
        elen = np.hypot(ex, ey)
        if elen < 1e-12:
            continue
        nx, ny = ey / elen, -ex / elen
        d1 = np.dot(A - B[k], [nx, ny])
        d2 = np.dot(B - B[k], [nx, ny])
        mn1, mx1 = d1.min(), d1.max()
        mn2, mx2 = d2.min(), d2.max()
        if mx1 < mn2 + EPS or mx2 < mn1 + EPS:
            return 0.0
        best = max(best, min(mx1, mx2) - max(mn1, mn2))
    return best


def _hex_verts_cache(params, side, n_inner):
    cx = params[0::3]
    cy = params[1::3]
    ang = params[2::3]
    r = side
    th = np.deg2rad(ang)[:, None] + np.arange(6) * np.pi / 3.0
    c = np.cos(th)
    s = np.sin(th)
    xs = cx[:, None] + r * c
    ys = cy[:, None] + r * s
    return np.stack((xs, ys), axis=-1)  # (n,6,2)


def _objective(params, S):
    n_inner = N
    cx = params[0::3]
    cy = params[1::3]
    ang = params[2::3]
    # containment: outer hexagon centered at origin, side S, flat-top
    outer = _hex_verts(0.0, 0.0, S, 0.0)
    m = outer.shape[0]
    pen = 0.0
    verts = _hex_verts_cache(params, 1.0, n_inner)
    for i in range(n_inner):
        vi = verts[i]
        for k in range(m):
            ax1, ay1 = outer[k]
            ax2, ay2 = outer[(k + 1) % m]
            ex, ey = ax2 - ax1, ay2 - ay1
            elen = np.hypot(ex, ey)
            if elen < 1e-12:
                continue
            nx, ny = ey / elen, -ex / elen
            d1 = np.dot(outer - outer[k], [nx, ny])
            d2 = np.dot(vi - outer[k], [nx, ny])
            mn1, mx1 = d1.min(), d1.max()
            mn2, mx2 = d2.min(), d2.max()
            if mx1 < mn2 + EPS or mx2 < mn1 + EPS:
                pen += 1e6
                break
            pen += (min(mx1, mx2) - max(mn1, mn2)) ** 2
    # pairwise disjointness
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            d = _poly_overlap_depth(verts[i], verts[j])
            if d > 0.0:
                pen += 10.0 * d * d
    return pen


def _fit(S, t0):
    """Try to find a valid packing of side S. Returns (params, ok) or None."""
    from scipy.optimize import minimize as _min
    best_pen = np.inf
    best_params = None
    n_inner = N
    for seed in range(6):
        if time.time() - t0 > DEADLINE - 12.0:
            break
        rng = np.random.default_rng(seed * 131 + 7)
        # initial positions: ring + center-ish, random angles
        ang0 = rng.uniform(0, 180, n_inner)
        base = []
        base.append([0.0, 0.0])
        for k in range(n_inner - 1):
            a = 2 * np.pi * k / (n_inner - 1)
            base.append([2.2 * np.cos(a), 2.2 * np.sin(a)])
        base = np.array(base)
        base[:, 0] += rng.uniform(-0.5, 0.5, n_inner)
        base[:, 1] += rng.uniform(-0.5, 0.5, n_inner)
        x0 = np.empty(3 * n_inner)
        x0[0::3] = base[:, 0]
        x0[1::3] = base[:, 1]
        x0[2::3] = ang0
        res = _min(_objective, x0, args=(S,), method="Nelder-Mead",
                   options={'maxiter': 2500, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
        if res.fun < best_pen:
            best_pen = res.fun
            best_params = res.x
    return best_params, best_pen < 1e-5


import time


def construct():
    t0 = time.time()
    # binary search on outer side length
    lo, hi = 3.0, 12.0
    best_params = None
    best_S = None
    # ensure hi feasible
    p, ok = _fit(hi, t0)
    if ok:
        best_params, best_S = p, hi
    for _ in range(14):
        if time.time() - t0 > DEADLINE - 12.0:
            break
        mid = 0.5 * (lo + hi)
        p, ok = _fit(mid, t0)
        if ok:
            hi = mid
            best_params, best_S = p, mid
        else:
            lo = mid
    if best_params is None:
        # fallback: seed-like layout
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    cx = best_params[0::3]
    cy = best_params[1::3]
    ang = best_params[2::3]
    inner = [[float(cx[i]), float(cy[i]), float(ang[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(best_S), 0.0
