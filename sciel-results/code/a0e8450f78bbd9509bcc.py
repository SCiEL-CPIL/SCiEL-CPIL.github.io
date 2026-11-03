import numpy as np
import time

N = 12
_TIME_LIMIT = 150.0
_DEADLINE = time.time() + _TIME_LIMIT - 8.0  # safety margin

# 12 unit hexagons in a 3-3-3-3 hexagonal lattice (optimal mutual packing, gap 0)
# row offsets: -1.5, -0.5, +0.5, +1.5 ; x offsets: -1,0,1
_ROWS = (-1.5, -0.5, 0.5, 1.5)
_COLS = (-1.0, 0.0, 1.0)
_LATTICE = [(c, r) for r in _ROWS for c in _COLS]  # 12 points


def _hex_vertices(cx, cy, s, ang_deg):
    """Vertices of a regular hexagon, side s, center (cx,cy), rotation ang_deg."""
    a = np.deg2rad(ang_deg)
    ca, sa = np.cos(a), np.sin(a)
    base = np.array([np.cos(np.deg2rad(60 * k)) for k in range(6)])
    basev = np.array([np.sin(np.deg2rad(60 * k)) for k in range(6)])
    pts = np.stack([base, basev], axis=1)
    R = np.array([[ca, -sa], [sa, ca]])
    return (pts @ R.T) * s + np.array([cx, cy])


def _min_outer_side(centers):
    """Minimal enclosing regular hexagon: returns (cx, cy, side, angle_deg).

    A regular hexagon of side s and rotation a contains a point p iff
    for all k in 0..5: (p-c)·u_k <= s, where u_k = R(a)·(cos60k, sin60k).
    So s_min = max_k (p-c)·u_k. Minimize s over (cx,cy,a) via Nelder-Mead.
    """
    centers = np.asarray(centers, dtype=float)

    def side_for(cx, cy, a):
        a = np.deg2rad(a)
        ca, sa = np.cos(a), np.sin(a)
        R = np.array([[ca, -sa], [sa, ca]])
        base = np.stack([np.cos(np.deg2rad(60 * k)),
                         np.sin(np.deg2rad(60 * k))], axis=1)
        U = base @ R.T  # 6x2
        d = (centers - np.array([cx, cy])) @ U.T  # 12x6
        return d.max()

    def obj(v):
        return side_for(v[0], v[1], v[2])

    # initial guess: centroid, angle 0
    c0 = centers.mean(axis=0)
    x0 = np.array([c0[0], c0[1], 0.0])

    from scipy.optimize import minimize
    best = x0
    bestf = obj(x0)
    t0 = time.time()
    for it in range(60):
        if time.time() > _DEADLINE:
            break
        res = minimize(obj, best, method='Nelder-Mead',
                       options={'maxiter': 3000, 'maxfev': 20000,
                                'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True})
        if res.fun < bestf:
            bestf = res.fun
            best = res.x
        # refine around best with a tighter start
        if bestf < 5.5:
            res2 = minimize(obj, best, method='Nelder-Mead',
                            options={'maxiter': 2000, 'maxfev': 10000,
                                     'xatol': 1e-10, 'fatol': 1e-12,
                                     'adaptive': True})
            if res2.fun < bestf:
                bestf = res2.fun
                best = res2.x
        if time.time() > _DEADLINE:
            break
    return best[0], best[1], bestf, np.rad2deg(best[2])


def construct():
    centers = [[c, r, 0.0] for (c, r) in _LATTICE]
    cx, cy, side, ang = _min_outer_side([[c, r] for (c, r) in _LATTICE])
    return centers, (cx, cy), side, ang
