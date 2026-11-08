import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    # Fourier modes for radial representation
    K_MAX = 6

    def _fourier_to_points(coeffs):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        r = np.ones(N)
        for k in range(1, K_MAX + 1):
            a_k = coeffs[2*(k-1)]
            b_k = coeffs[2*(k-1)+1]
            r += a_k * np.cos(k * angles) + b_k * np.sin(k * angles)
        r = np.maximum(r, 0.05)
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        return np.column_stack([xs, ys])

    def compute_score(points):
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _soft_objective(coeffs, beta=50.0):
        pts = _fourier_to_points(coeffs)
        pts = pts - pts.mean(axis=0)
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            return 1e6
        if hull_area < 1e-9:
            return 1e6
        normalized_areas = areas / hull_area
        neg_beta_areas = -beta * normalized_areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_ratio = -log_sum_exp / beta
        return -soft_min_ratio

    def _hard_objective(coeffs):
        pts = _fourier_to_points(coeffs)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _cartesian_soft_objective(points_flat, beta=50.0):
        pts = points_flat.reshape(-1, 2)
        pts = pts - pts.mean(axis=0)
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            return 1e6
        if hull_area < 1e-9:
            return 1e6
        normalized_areas = areas / hull_area
        neg_beta_areas = -beta * normalized_areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_ratio = -log_sum_exp / beta
        return -soft_min_ratio

    def _cartesian_hard_objective(points_flat):
        pts = points_flat.reshape(-1, 2)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    best_score = -float('inf')
    best_points = None
    n_coeffs = 2 * K_MAX
    rng = np.random.RandomState(42)

    # Fourier coefficient initializations
    init_coeffs_list = [np.zeros(n_coeffs)]
    for scale in [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]:
        for trial in range(4):
            init_coeffs_list.append(rng.randn(n_coeffs) * scale)
    for k in range(1, K_MAX + 1):
        for amp in [0.03, 0.07, 0.12]:
            c = np.zeros(n_coeffs)
            c[2*(k-1)] = amp
            init_coeffs_list.append(c)
            c2 = np.zeros(n_coeffs)
            c2[2*(k-1)+1] = amp
            init_coeffs_list.append(c2)
    for k1 in range(1, K_MAX + 1):
        for k2 in range(k1 + 1, K_MAX + 1):
            for amp in [0.05, 0.1]:
                c = np.zeros(n_coeffs)
                c[2*(k1-1)] = amp
                c[2*(k2-1)] = amp
                init_coeffs_list.append(c)

    # Optimize Fourier initializations
    for init_coeffs in init_coeffs_list:
        if time.time() > start_time + time_limit - 25:
            break
        current = init_coeffs.copy()
        betas = [5.0, 15.0, 40.0, 100.0, 250.0]
        for beta in betas:
            if time.time() > start_time + time_limit - 12:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current,
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                current = res.x
            except Exception:
                pass
        if time.time() > start_time + time_limit - 8:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current,
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            candidate_coeffs = res.x
            candidate_points = _fourier_to_points(candidate_coeffs)
            score = compute_score(candidate_points)
            if score > best_score:
                best_score = score
                best_points = candidate_points.copy()
        except Exception:
            pass

    # Cartesian refinement of best Fourier result
    if best_points is not None and time.time() < start_time + time_limit - 5:
        try:
            res = opt.minimize(
                lambda x, b=50.0: _cartesian_soft_objective(x, b),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 60, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            refined = res.x.reshape(-1, 2)
            score = compute_score(refined)
            if score > best_score:
                best_score = score
                best_points = refined.copy()
        except Exception:
            pass

    # Final Nelder-Mead polish in Cartesian
    if best_points is not None and time.time() < start_time + time_limit - 2:
        try:
            res = opt.minimize(
                _cartesian_hard_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-14}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass

    if best_points is None or best_score < 0.001:
        best_points = _regular_polygon(13, 1.0)
        best_score = compute_score(best_points)

    return best_points
