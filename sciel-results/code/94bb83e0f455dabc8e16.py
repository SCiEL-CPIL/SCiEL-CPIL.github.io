import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin for safety

    # Precompute triple index arrays for vectorized area computation
    _tri_i = []
    _tri_j = []
    _tri_k = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                _tri_i.append(i)
                _tri_j.append(j)
                _tri_k.append(k)
    tri_i = np.array(_tri_i, dtype=np.int32)
    tri_j = np.array(_tri_j, dtype=np.int32)
    tri_k = np.array(_tri_k, dtype=np.int32)
    n_triples = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area for given points."""
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
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        min_area = float(np.min(areas))
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points so convex hull has unit area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _soft_objective(points_flat, beta=50.0):
        """
        Normalized soft-min objective. Points are centered and scaled to unit hull area.
        Returns negative soft-min triangle area (since hull area = 1, score = min_area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        x = pts[:, 0]
        y = pts[:, 1]
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
        return -soft_min_area

    def _hard_objective(points_flat):
        """Negative hard score on normalized points."""
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        score = compute_score(pts)
        if score <= 0:
            return 1e6
        return -score

    def _coordinate_descent(pts, n_rounds=3, time_budget=None):
        """
        Iteratively optimize each point (2D) while fixing others, on normalized space.
        This is a different search paradigm from gradient-based methods.
        """
        best = pts.copy()
        best_s = compute_score(best)
        for rnd in range(n_rounds):
            if time_budget is not None and time.time() > time_budget:
                break
            improved = False
            for i in range(N):
                if time_budget is not None and time.time() > time_budget:
                    break
                # Optimize point i
                fixed = best.copy()
                def obj_i(x2):
                    cand = fixed.copy()
                    cand[i] = x2
                    return _hard_objective(cand.flatten())
                try:
                    res = opt.minimize(
                        obj_i, best[i].copy(), method='Nelder-Mead',
                        options={'maxiter': 40, 'xatol': 1e-9, 'fatol': 1e-11}
                    )
                    cand = best.copy()
                    cand[i] = res.x
                    s = compute_score(cand)
                    if s > best_s + 1e-12:
                        best_s = s
                        best = cand
                        improved = True
                except Exception:
                    pass
            if not improved:
                break
        return best, best_s

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _ring_config(r1, r2, phase1=0.0, phase2=0.0, n1=0, n2=0, n3=0):
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
            a = np.linspace(0, 2 * np.pi, n3, endpoint=False)
            r3 = 0.25 * r2 if r2 > 0 else 0.25
            pts.append(np.column_stack([r3 * np.cos(a), r3 * np.sin(a)]))
        if not pts:
            return _regular_polygon(N, radius=1.0)
        return np.vstack(pts)

    def _optimize_from(start_pts, time_budget):
        """
        Multi-stage optimization from a starting configuration.
        Uses normalized objectives throughout.
        """
        best = start_pts.copy()
        best_s = compute_score(best)
        current = start_pts.copy()

        # Stage 1: L-BFGS-B with annealed soft-min beta (normalized)
        for beta in [10.0, 25.0, 60.0, 120.0, 250.0]:
            if time.time() > time_budget:
                break
            try:
                res = opt.minimize(
                    lambda x: _soft_objective(x, beta),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                cand = res.x.reshape(-1, 2)
                s = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
                current = cand
            except Exception:
                pass

        # Stage 2: Powell with hard objective (normalized)
        if time.time() < time_budget - 2:
            try:
                res = opt.minimize(
                    _hard_objective,
                    best.flatten(),
                    method='Powell',
                    options={'maxiter': 80, 'xtol': 1e-12, 'ftol': 1e-12}
                )
                cand = res.x.reshape(-1, 2)
                s = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

        # Stage 3: Coordinate descent (different search paradigm)
        if time.time() < time_budget - 1:
            try:
                cd_best, cd_s = _coordinate_descent(best, n_rounds=2, time_budget=time_budget - 1)
                if cd_s > best_s:
                    best_s = cd_s
                    best = cd_best.copy()
            except Exception:
                pass

        # Stage 4: Nelder-Mead final polish
        if time.time() < time_budget:
            try:
                res = opt.minimize(
                    _hard_objective,
                    best.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 150, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                cand = res.x.reshape(-1, 2)
                s = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

        return best, best_s

    best_points = None
    best_score = -float('inf')

    # --- Phase 1: Diverse symmetric initializations ---
    ring_starts = [
        _regular_polygon(13, radius=1.0),
        np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]),
        _ring_config(r1=1.0, r2=0.5, n1=6, n2=6, n3=1),
        _ring_config(r1=1.0, r2=0.45, n1=7, n2=5, n3=1),
        _ring_config(r1=1.0, r2=0.4, n1=8, n2=4, n3=1),
        np.vstack([_regular_polygon(11, 1.0), np.array([[-0.4, 0.0], [0.4, 0.0]])]),
        np.vstack([_regular_polygon(10, 1.0), _regular_polygon(3, 0.5)]),
        np.vstack([_regular_polygon(9, 1.0), _regular_polygon(4, 0.5)]),
        np.vstack([_regular_polygon(8, 1.0), _regular_polygon(5, 0.5)]),
        np.vstack([_regular_polygon(7, 1.0), _regular_polygon(6, 0.5)]),
        np.vstack([_regular_polygon(6, 1.0), np.array([[0.0, 0.0]]), _regular_polygon(6, 0.5)]),
        np.vstack([_regular_polygon(5, 1.0), _regular_polygon(8, 0.5)]),
    ]

    # Add perturbed versions to break symmetry
    rng = np.random.RandomState(42)
    for i in range(0, len(ring_starts), 2):
        perturbed = ring_starts[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.05
        ring_starts.append(perturbed)

    n_starts = len(ring_starts)
    search_budget = time_limit - 15.0
    per_start = search_budget / n_starts

    for idx, start_pts in enumerate(ring_starts):
        if time.time() - start_time > time_limit - 15:
            break
        alloc = per_start * (1.2 if idx < 12 else 1.0)
        s = compute_score(start_pts)
        if s > best_score:
            best_score = s
            best_points = start_pts.copy()

        cand, cand_s = _optimize_from(start_pts, start_time + alloc)
        if cand_s > best_score:
            best_score = cand_s
            best_points = cand.copy()

    # --- Phase 2: Final high-precision polish ---
    if best_points is not None:
        remaining = time_limit - 5.0 - (time.time() - start_time)
        if remaining > 8:
            try:
                res = opt.minimize(
                    lambda x: _soft_objective(x, 300.0),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                cand = res.x.reshape(-1, 2)
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

        remaining = time_limit - 3.0 - (time.time() - start_time)
        if remaining > 5:
            try:
                res = opt.minimize(
                    _hard_objective,
                    best_points.flatten(),
                    method='Powell',
                    options={'maxiter': 100, 'xtol': 1e-13, 'ftol': 1e-13}
                )
                cand = res.x.reshape(-1, 2)
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

        remaining = time_limit - 2.0 - (time.time() - start_time)
        if remaining > 3:
            try:
                cd_best, cd_s = _coordinate_descent(best_points, n_rounds=3, time_budget=time_limit - 2.0)
                if cd_s > best_score:
                    best_score = cd_s
                    best_points = cd_best.copy()
            except Exception:
                pass

        remaining = time_limit - 1.0 - (time.time() - start_time)
        if remaining > 2:
            try:
                res = opt.minimize(
                    _hard_objective,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                cand = res.x.reshape(-1, 2)
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass

    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)

    return best_points

# EVOLVE-BLOCK-END
