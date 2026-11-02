import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13
TIME_LIMIT = 145.0

def construct():
    start_time = time.time()
    time_limit = TIME_LIMIT

    # Precompute triple indices
    tri_i, tri_j, tri_k = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                tri_i.append(i)
                tri_j.append(j)
                tri_k.append(k)
    tri_i = np.array(tri_i, dtype=np.int32)
    tri_j = np.array(tri_j, dtype=np.int32)
    tri_k = np.array(tri_k, dtype=np.int32)
    n_triples = len(tri_i)

    def compute_score(points):
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0
        if area_hull < 1e-9:
            return 0.0, 0.0
        x, y = points[:, 0], points[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        min_area = float(np.min(areas))
        score = min_area / area_hull
        return score, area_hull

    def _soft_min_score(points, beta=50.0):
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6
        if area_hull < 1e-9:
            return 1e6
        x, y = points[:, 0], points[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        return -(soft_min_area / area_hull)

    def _soft_min_grad(points, beta=50.0):
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return None, None
        if area_hull < 1e-9:
            return None, None
        x, y = points[:, 0], points[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = v1x * v2y - v1y * v2x
        areas = 0.5 * np.abs(cross)
        sign_cross = np.sign(cross)
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        exp_vals = np.exp(neg_beta_areas - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        coeff = -weights * sign_cross
        d_soft_min_dx = np.zeros(N)
        d_soft_min_dy = np.zeros(N)
        d_soft_min_dx[tri_i] += coeff * (y[tri_j] - y[tri_k])
        d_soft_min_dx[tri_j] += coeff * (y[tri_k] - y[tri_i])
        d_soft_min_dx[tri_k] += coeff * (-(y[tri_j] - y[tri_i]))
        d_soft_min_dy[tri_i] += coeff * (x[tri_j] - x[tri_k])
        d_soft_min_dy[tri_j] += coeff * (-(x[tri_k] - x[tri_i]))
        d_soft_min_dy[tri_k] += coeff * (x[tri_j] - x[tri_i])
        grad_x = -d_soft_min_dx / area_hull
        grad_y = -d_soft_min_dy / area_hull
        return grad_x, grad_y

    def _epigraph_objective(params):
        """
        Epigraph formulation: maximize t subject to area_t >= t * hull_area for all t.
        params = [x_0, y_0, ..., x_12, y_12, t]
        We minimize -t + penalty for violated constraints.
        """
        points = params[:2*N].reshape(N, 2)
        t = params[2*N]
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6
        if area_hull < 1e-9:
            return 1e6
        x, y = points[:, 0], points[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        # Constraint: areas >= t * area_hull
        violations = np.maximum(0, t * area_hull - areas)
        penalty = 1e4 * np.sum(violations ** 2)
        return -t + penalty

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _ring_config(r1, r2, r3, n1, n2, n3, phase1=0.0, phase2=0.0, phase3=0.0):
        pts = []
        if n1 > 0:
            a = np.linspace(phase1, phase1 + 2 * np.pi, n1, endpoint=False)
            pts.append(np.column_stack([r1 * np.cos(a), r1 * np.sin(a)]))
        if n2 > 0:
            a = np.linspace(phase2, phase2 + 2 * np.pi, n2, endpoint=False)
            pts.append(np.column_stack([r2 * np.cos(a), r2 * np.sin(a)]))
        if n3 == 1:
            pts.append(np.array([[0.0, 0.0]]))
        elif n3 > 1:
            a = np.linspace(phase3, phase3 + 2 * np.pi, n3, endpoint=False)
            pts.append(np.column_stack([r3 * np.cos(a), r3 * np.sin(a)]))
        if not pts:
            return _regular_polygon(N, radius=1.0)
        return np.vstack(pts)

    def _optimize_from(start_pts, time_budget):
        best = start_pts.copy()
        best_s, _ = compute_score(best)
        current = start_pts.copy()
        elapsed = time.time() - start_time

        # Stage 1: L-BFGS-B with annealed soft-min beta
        for beta in [8.0, 20.0, 50.0, 100.0, 200.0, 400.0]:
            if time.time() - start_time > time_budget:
                break
            try:
                res = opt.minimize(
                    lambda x: _soft_min_score(x.reshape(-1, 2), beta),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 60}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
                current = cand
            except Exception:
                pass

        # Stage 2: Epigraph formulation with SLSQP
        if time.time() - start_time < time_budget - 10:
            try:
                # Initialize t from current best
                s_init, _ = compute_score(current)
                x0 = np.concatenate([current.flatten(), [s_init]])
                bounds = [(-2.0, 2.0)] * (2 * N) + [(0.0, 0.5)]
                res = opt.minimize(
                    _epigraph_objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-12}
                )
                cand = res.x[:2*N].reshape(N, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
                current = cand
            except Exception:
                pass

        # Stage 3: Powell method
        if time.time() - start_time < time_budget - 5:
            try:
                res = opt.minimize(
                    lambda x: -compute_score(x.reshape(-1, 2))[0],
                    best.flatten(),
                    method='Powell',
                    options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-10}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

        # Stage 4: Nelder-Mead polish
        if time.time() - start_time < time_budget - 3:
            try:
                res = opt.minimize(
                    lambda x: -compute_score(x.reshape(-1, 2))[0],
                    best.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-11, 'fatol': 1e-13}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

        return best, best_s

    best_points = None
    best_score = -float('inf')

    # Ring starts - focus on promising configurations
    ring_starts = [
        _ring_config(r1=1.0, r2=0.0, r3=0.0, n1=12, n2=0, n3=1),
        _ring_config(r1=1.0, r2=0.5, r3=0.0, n1=6, n2=6, n3=1),
        _ring_config(r1=1.0, r2=0.45, r3=0.0, n1=7, n2=5, n3=1),
        _ring_config(r1=1.0, r2=0.4, r3=0.0, n1=8, n2=4, n3=1),
        _ring_config(r1=1.0, r2=0.55, r3=0.0, n1=7, n2=5, n3=1, phase2=np.pi/3),
        _ring_config(r1=1.0, r2=0.5, r3=0.0, n1=8, n2=4, n3=1, phase2=np.pi/4),
        _ring_config(r1=1.0, r2=0.6, r3=0.0, n1=9, n2=3, n3=1),
        _ring_config(r1=1.0, r2=0.35, r3=0.0, n1=9, n2=3, n3=1, phase2=np.pi/6),
        _ring_config(r1=1.0, r2=0.45, r3=0.0, n1=10, n2=2, n3=1, phase2=np.pi/5),
        _ring_config(r1=1.0, r2=0.65, r3=0.0, n1=8, n2=4, n3=1, phase2=np.pi/7),
        _ring_config(r1=1.0, r2=0.5, r3=0.0, n1=11, n2=1, n3=1, phase2=np.pi/2),
        _ring_config(r1=1.0, r2=0.3, r3=0.0, n1=11, n2=2, n3=0, phase2=np.pi/2),
        _ring_config(r1=1.0, r2=0.4, r3=0.0, n1=10, n2=3, n3=0, phase2=np.pi/3),
        _ring_config(r1=1.0, r2=0.5, r3=0.0, n1=6, n2=6, n3=1, phase1=np.pi/12),
        _ring_config(r1=1.0, r2=0.45, r3=0.0, n1=7, n2=5, n3=1, phase1=np.pi/7),
    ]

    n_starts = len(ring_starts)
    search_budget = time_limit - 15.0
    per_start = search_budget / n_starts

    for idx, start_pts in enumerate(ring_starts):
        if time.time() > start_time + time_limit - 15:
            break
        remaining = time_limit - 15 - (time.time() - start_time)
        alloc = min(per_start * 1.2, remaining)
        if alloc < 3.0:
            break
        cand, cand_s = _optimize_from(start_pts, time.time() - start_time + alloc)
        if cand_s > best_score:
            best_score = cand_s
            best_points = cand.copy()

    # Final high-precision polish
    if best_points is not None and time.time() < start_time + time_limit - 12:
        # High-beta soft-min
        try:
            result = opt.minimize(
                lambda x: _soft_min_score(x.reshape(-1, 2), 600.0),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100}
            )
            polished = result.x.reshape(-1, 2)
            pol_score, _ = compute_score(polished)
            if pol_score > best_score:
                best_score = pol_score
                best_points = polished.copy()
        except Exception:
            pass

        # Epigraph polish
        if time.time() < start_time + time_limit - 10:
            try:
                s_init, _ = compute_score(best_points)
                x0 = np.concatenate([best_points.flatten(), [s_init]])
                bounds = [(-2.0, 2.0)] * (2 * N) + [(0.0, 0.5)]
                res = opt.minimize(
                    _epigraph_objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-13}
                )
                cand = res.x[:2*N].reshape(N, 2)
                s, _ = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

        # Powell refinement
        if time.time() < start_time + time_limit - 8:
            try:
                res = opt.minimize(
                    lambda x: -compute_score(x.reshape(-1, 2))[0],
                    best_points.flatten(),
                    method='Powell',
                    options={'maxiter': 300, 'ftol': 1e-13, 'xtol': 1e-11}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

        # Final NM polish
        if time.time() < start_time + time_limit - 5:
            try:
                res = opt.minimize(
                    lambda x: -compute_score(x.reshape(-1, 2))[0],
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 600, 'adaptive': True, 'xatol': 1e-13, 'fatol': 1e-15}
                )
                final_points = res.x.reshape(-1, 2)
                final_score, _ = compute_score(final_points)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
            except Exception:
                pass

    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)

    return best_points
