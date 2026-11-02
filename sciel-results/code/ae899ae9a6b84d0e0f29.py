import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin below the 150s kill

    def _elapsed():
        return time.time() - start_time

    def _remaining():
        return time_limit - _elapsed()

    # ---- fast convex-hull area (no scipy overhead) ----
    def _hull_area(pts):
        n = pts.shape[0]
        if n < 3:
            return 0.0
        p = pts.copy()
        p[np.argsort(p[:, 0])] = 0  # no-op guard
        # sort by x then y
        order = np.lexsort((p[:, 1], p[:, 0]))
        p = p[order]
        # Andrew monotone chain
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lower = []
        for pt in p:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
                lower.pop()
            lower.append(pt)
        upper = []
        for pt in p[::-1]:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
                upper.pop()
            upper.append(pt)
        hull = np.array(lower[:-1] + upper[:-1])
        if len(hull) < 3:
            return 0.0
        x = hull[:, 0]; y = hull[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    # ---- vectorized min triangle area over all triples ----
    def _min_tri_area(pts):
        x = pts[:, 0]; y = pts[:, 1]
        n = pts.shape[0]
        i = np.repeat(np.arange(n), n * (n - 1) // 2)
        j = np.tile(np.repeat(np.arange(n), n - 1), n)
        k = np.tile(np.tile(np.arange(n), n - 1), n)
        mask = (i < j) & (j < k)
        i, j, k = i[mask], j[mask], k[mask]
        cross = (x[j] - x[i]) * (y[k] - y[i]) - (y[j] - y[i]) * (x[k] - x[i])
        return 0.5 * np.min(np.abs(cross))

    def _score(pts):
        ha = _hull_area(pts)
        if ha <= 1e-12:
            return 0.0
        return _min_tri_area(pts) / ha

    # ---- smooth soft-min surrogate (log-sum-exp) for gradient methods ----
    def _neg_soft(pts_flat, beta=40.0):
        pts = pts_flat.reshape(-1, 2)
        ha = _hull_area(pts)
        if ha <= 1e-12:
            return 1e6
        x = pts[:, 0]; y = pts[:, 1]
        n = pts.shape[0]
        i = np.repeat(np.arange(n), n * (n - 1) // 2)
        j = np.tile(np.repeat(np.arange(n), n - 1), n)
        k = np.tile(np.tile(np.arange(n), n - 1), n)
        mask = (i < j) & (j < k)
        i, j, k = i[mask], j[mask], k[mask]
        cross = (x[j] - x[i]) * (y[k] - y[i]) - (y[j] - y[i]) * (x[k] - x[i])
        areas = 0.5 * np.abs(cross)
        z = -beta * areas
        m = np.max(z)
        soft_min = -(m + np.log(np.sum(np.exp(z - m)))) / beta
        return -(soft_min / ha)

    def _neg_hard(pts_flat):
        return -_score(pts_flat.reshape(-1, 2))

    # ---- symmetric layered parameterization ----
    def _build(params):
        outer_n = int(np.rint(params[0]))
        outer_n = min(max(outer_n, 3), N - 1)
        inner_n = N - outer_n
        r_out = params[1]
        r_in = params[2]
        rot = params[3]
        inner_rot = params[4]
        pts = []
        if outer_n > 0:
            a = np.linspace(0, 2 * np.pi, outer_n, endpoint=False) + rot
            pts.append(np.column_stack([r_out * np.cos(a), r_out * np.sin(a)]))
        if inner_n > 0:
            a = np.linspace(0, 2 * np.pi, inner_n, endpoint=False) + inner_rot
            pts.append(np.column_stack([r_in * np.cos(a), r_in * np.sin(a)]))
        return np.vstack(pts)

    def _neg_soft_layered(params):
        return _neg_soft(_build(params).flatten())

    def _neg_hard_layered(params):
        return _neg_hard(_build(params).flatten())

    best_points = None
    best_score = -float('inf')

    def _consider(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s

    # ---- Phase A: global grid search over symmetric layers ----
    best_layer_params = None
    for outer_n in range(4, N):  # 4..12 outer
        if _remaining() < 3.0:
            break
        for r_out in [0.9, 1.0, 1.1]:
            for r_in in [0.0, 0.2, 0.35, 0.5, 0.65]:
                if _remaining() < 1.0:
                    break
                for rot in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    for inner_rot in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                        params = np.array([float(outer_n), r_out, r_in, rot, inner_rot])
                        pts = _build(params)
                        s = _score(pts)
                        if s > best_score:
                            best_score = s
                            best_points = pts.copy()
                            best_layer_params = params
    if _remaining() < 5.0:
        if best_points is None:
            a = np.linspace(0, 2 * np.pi, N, endpoint=False)
            best_points = np.column_stack([np.cos(a), np.sin(a)])
        return best_points

    # ---- Phase B: local refine of best symmetric config ----
    if best_layer_params is not None and _remaining() > 8.0:
        for beta in [20.0, 50.0, 100.0]:
            if _remaining() < 6.0:
                break
            try:
                res = opt.minimize(
                    lambda p: _neg_soft_layered(p),
                    best_layer_params,
                    method='L-BFGS-B',
                    options={'maxiter': 60, 'ftol': 1e-12, 'gtol': 1e-10},
                )
                pts = _build(res.x)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
                    best_layer_params = res.x.copy()
            except Exception:
                pass
    # ---- Phase C: Nelder-Mead on the full point set (hard score) ----
    if best_points is not None and _remaining() > 10.0:
        try:
            res = opt.minimize(
                _neg_hard,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 400, 'adaptive': True, 'xatol': 1e-9, 'fatol': 1e-11},
            )
            pts = res.x.reshape(-1, 2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass

    # ---- Phase D: a couple of random restarts with soft-min polish ----
    if _remaining() > 12.0:
        rng = np.random.default_rng(123)
        for _ in range(4):
            if _remaining() < 8.0:
                break
            outer_n = int(rng.integers(5, 12))
            inner_n = N - outer_n
            rot = rng.uniform(0, 2 * np.pi)
            inner_rot = rng.uniform(0, 2 * np.pi)
            params = np.array([float(outer_n),
                               rng.uniform(0.85, 1.15),
                               rng.uniform(0.2, 0.6) if inner_n > 0 else 0.0,
                               rot, inner_rot])
            pts = _build(params)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
            if _remaining() < 6.0:
                break
            try:
                res = opt.minimize(
                    lambda p: _neg_soft_layered(p),
                    params,
                    method='L-BFGS-B',
                    options={'maxiter': 40, 'ftol': 1e-12, 'gtol': 1e-10},
                )
                pts = _build(res.x)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
            except Exception:
                pass

    if best_points is None:
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(a), np.sin(a)])

    return best_points

# EVOLVE-BLOCK-END
