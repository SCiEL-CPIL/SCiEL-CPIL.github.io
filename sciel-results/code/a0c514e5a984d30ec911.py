import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 143.0  # 7s safety margin before the 150s kill

    # --- Precompute the 286 triple index arrays (vectorized area) ---
    _ti, _tj, _tk = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                _ti.append(i); _tj.append(j); _tk.append(k)
    ti = np.asarray(_ti, dtype=np.int32)
    tj = np.asarray(_tj, dtype=np.int32)
    tk = np.asarray(_tk, dtype=np.int32)

    def _triangle_areas(pts):
        x = pts[:, 0]; y = pts[:, 1]
        v1x = x[tj] - x[ti]; v1y = y[tj] - y[ti]
        v2x = x[tk] - x[ti]; v2y = y[tk] - y[ti]
        return 0.5 * np.abs(v1x * v2y - v1y * v2x)

    def compute_score(pts):
        try:
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0
        if area_hull < 1e-9:
            return 0.0, 0.0
        min_area = float(np.min(_triangle_areas(pts)))
        return min_area / area_hull, area_hull

    def soft_obj(pts, beta):
        try:
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 1e6
        if area_hull < 1e-9:
            return 1e6
        areas = _triangle_areas(pts)
        z = -beta * areas
        m = np.max(z)
        if m < -1e6:
            return 1e6
        lse = m + np.log(np.sum(np.exp(z - m)))
        return -(-lse / beta) / area_hull

    def _regular(n, radius, center, phase):
        if n == 1:
            return np.array([center])
        a = phase + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.column_stack([center[0] + radius * np.cos(a),
                                center[1] + radius * np.sin(a)])

    # --- Symmetric model: 1 center + 6 mid (phase pm) + 6 outer (phase po) ---
    def sym_points(p):
        r_out = float(p[0]); r_mid = float(p[1]); pm = float(p[2]); po = float(p[3])
        if r_out < 0.05 or r_mid < 0.02 or r_mid >= 0.99 * r_out:
            return None
        outer = _regular(6, r_out, (0.0, 0.0), po)
        mid = _regular(6, r_mid, (0.0, 0.0), pm)
        return np.vstack([np.array([[0.0, 0.0]]), mid, outer])

    def sym_soft(p, beta):
        pts = sym_points(p)
        if pts is None:
            return 1e6
        return soft_obj(pts, beta)

    best_pts = None
    best_score = -1e30

    def _consider(pts):
        nonlocal best_pts, best_score
        if pts is None:
            return
        s, _ = compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # --- Symmetric optimization (primary, high-yield) ---
    sym_starts = [
        [1.0, 0.55, np.pi / 6.0, 0.0],
        [1.0, 0.55, 0.0, 0.0],
        [1.0, 0.50, np.pi / 6.0, 0.0],
        [1.0, 0.60, np.pi / 6.0, 0.0],
        [1.0, 0.48, 0.0, 0.0],
        [1.0, 0.55, 0.3, 0.0],
        [1.0, 0.55, -0.3, 0.0],
        [1.0, 0.55, 0.5, 0.0],
    ]
    for s in sym_starts:
        _consider(sym_points(s))

    for s in sym_starts:
        if time.time() - start_time > time_limit - 12.0:
            break
        cur = np.array(s, dtype=float)
        for beta in [15.0, 40.0, 100.0, 250.0]:
            if time.time() - start_time > time_limit - 12.0:
                break
            try:
                res = opt.minimize(lambda x: sym_soft(x, beta), cur,
                                   method='L-BFGS-B',
                                   options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-13})
                cur = res.x
            except Exception:
                pass
        _consider(sym_points(cur))
        # hard Nelder-Mead polish in param space
        if time.time() - start_time < time_limit - 10.0:
            try:
                res = opt.minimize(
                    lambda x: -compute_score(sym_points(x))[0] if sym_points(x) is not None else 1e6,
                    cur, method='Nelder-Mead',
                    options={'maxiter': 300, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14})
                _consider(sym_points(res.x))
            except Exception:
                pass

    # --- Generic full-space fallback (only if symmetric result is degenerate) ---
    if best_score < 1.0:
        def gen_soft(p, beta):
            pts = p.reshape(-1, 2)
            return soft_obj(pts, beta)

        generic_starts = [
            _regular(13, 1.0, (0.0, 0.0), 0.0),
            np.vstack([np.array([[0.0, 0.0]]),
                       _regular(6, 0.55, (0.0, 0.0), np.pi / 6.0),
                       _regular(6, 1.0, (0.0, 0.0), 0.0)]),
        ]
        for g in generic_starts:
            if time.time() - start_time > time_limit - 10.0:
                break
            cur = g.copy()
            for beta in [20.0, 60.0, 150.0]:
                if time.time() - start_time > time_limit - 10.0:
                    break
                try:
                    res = opt.minimize(lambda x: gen_soft(x, beta), cur.flatten(),
                                       method='L-BFGS-B',
                                       options={'maxiter': 80, 'ftol': 1e-15, 'gtol': 1e-13})
                    cur = res.x.reshape(-1, 2)
                except Exception:
                    pass
            _consider(cur)

    # --- Final high-precision polish on the best point found ---
    if best_pts is not None:
        remaining = time_limit - 4.0 - (time.time() - start_time)
        if remaining > 6.0:
            try:
                res = opt.minimize(
                    lambda x: -compute_score(x.reshape(-1, 2))[0],
                    best_pts.flatten(), method='Nelder-Mead',
                    options={'maxiter': 400, 'adaptive': True, 'xatol': 1e-13, 'fatol': 1e-15})
                _consider(res.x.reshape(-1, 2))
            except Exception:
                pass

    if best_pts is None:
        best_pts = _regular(13, 1.0, (0.0, 0.0), 0.0)

    return best_pts
# EVOLVE-BLOCK-END
