import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 142.0  # Leave 8s margin before the 150s kill

    # Precompute triple index arrays for vectorized area computation
    _ti, _tj, _tk = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                _ti.append(i); _tj.append(j); _tk.append(k)
    tri_i = np.array(_ti, dtype=np.int32)
    tri_j = np.array(_tj, dtype=np.int32)
    tri_k = np.array(_tk, dtype=np.int32)
    n_triples = len(tri_i)

    def _hull_area(points):
        try:
            h = ConvexHull(points)
            return h.volume
        except Exception:
            return 0.0

    def compute_score(points):
        """Return (min_triangle_area / hull_area, hull_area)."""
        area_hull = _hull_area(points)
        if area_hull < 1e-9:
            return 0.0, 0.0
        x = points[:, 0]; y = points[:, 1]
        v1x = x[tri_j] - x[tri_i]; v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]; v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        return float(np.min(areas)) / area_hull, area_hull

    def _soft_min_score(points, beta=50.0, area_ref=1.0):
        """
        Smooth surrogate for the min-triangle-area / hull-area ratio.
        Returns a value to MINIMIZE:  -soft_ratio  +  penalty*(1 - area/area_ref)^2
        """
        area_hull = _hull_area(points)
        if area_hull < 1e-9:
            return 1e6
        x = points[:, 0]; y = points[:, 1]
        v1x = x[tri_j] - x[tri_i]; v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]; v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)

        neg = -beta * areas
        m = np.max(neg)
        if m < -1e6:
            return 1e6
        soft_min = -(m + np.log(np.sum(np.exp(neg - m)))) / beta
        soft_ratio = soft_min / area_hull

        # Hull-area normalization penalty
        pen = 0.5 * (1.0 - area_hull / area_ref) ** 2
        return -soft_ratio + pen

    def _soft_objective(points_flat, beta=50.0, area_ref=1.0):
        return _soft_min_score(points_flat.reshape(-1, 2), beta, area_ref)

    def _hard_score_neg(points_flat):
        s, _ = compute_score(points_flat.reshape(-1, 2))
        return -s

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), phase=0.0):
        if n == 1:
            return np.array([center, center])
        a = np.linspace(phase, phase + 2 * np.pi, n, endpoint=False)
        return np.column_stack([center[0] + radius * np.cos(a),
                                center[1] + radius * np.sin(a)])

    def _ring_config(r1, n1, r2=0.0, n2=0, n3=0, phase1=0.0, phase2=0.0, r3=0.0, phase3=0.0):
        """n1 outer + n2 middle + n3 inner (or 1 center) = N points."""
        parts = []
        if n1 > 0:
            parts.append(_regular_polygon(n1, r1, phase=phase1))
        if n2 > 0:
            parts.append(_regular_polygon(n2, r2, phase=phase2))
        if n3 == 1:
            parts.append(np.array([[0.0, 0.0]]))
        elif n3 > 1:
            parts.append(_regular_polygon(n3, r3 if r3 > 0 else 0.25 * max(r2, 0.3), phase=phase3))
        if not parts:
            return _regular_polygon(N, 1.0)
        return np.vstack(parts)

    def _optimize_from(start_pts, budget_deadline):
        """Multi-stage refine: annealed soft-min L-BFGS-B, then NM hard polish, then Powell."""
        best = start_pts.copy()
        best_s, _ = compute_score(best)
        current = start_pts.copy()
        area_ref = max(_hull_area(best), 1e-6)

        # Stage 1: annealed soft-min with progressive beta (start low, increase)
        betas = [8.0, 20.0, 50.0, 100.0, 200.0, 400.0]
        for beta in betas:
            if time.time() > budget_deadline:
                break
            try:
                res = opt.minimize(
                    lambda x: _soft_objective(x, beta, area_ref),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 80},
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s; best = cand.copy()
                current = cand
            except Exception:
                pass

        # Stage 2: Nelder-Mead polish on the hard objective
        if time.time() < budget_deadline - 2:
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    best.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-11, 'fatol': 1e-13},
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s; best = cand.copy()
            except Exception:
                pass

        # Stage 3: Powell refinement if time remains
        if time.time() < budget_deadline - 1:
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    best.flatten(),
                    method='Powell',
                    options={'maxiter': 300, 'xtol': 1e-10, 'ftol': 1e-12},
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s; best = cand.copy()
            except Exception:
                pass

        return best, best_s

    # ---- Build diverse starting configurations ----
    starts = []
    # Symmetric ring configs (strong geometry prior)
    starts.append(_ring_config(1.0, 12, n3=1))                              # 1 center + 12 outer
    starts.append(_ring_config(1.0, 6, 0.5, 6, 1))                          # 1 center + 6 mid + 6 outer
    starts.append(_ring_config(1.0, 7, 0.45, 5, 1))                         # 1 center + 5 mid + 7 outer
    starts.append(_ring_config(1.0, 8, 0.4, 4, 1))                          # 1 center + 4 mid + 8 outer
    starts.append(_ring_config(1.0, 5, 0.45, 7, 1))                         # 1 center + 7 mid + 5 outer
    starts.append(_ring_config(1.0, 13, 0.0, 0, 0))                         # pure 13-gon
    starts.append(_ring_config(1.0, 12, n3=1, phase1=np.pi / 12))           # phase-shifted 12 outer
    starts.append(_ring_config(1.0, 9, 0.6, 3, 1))                          # 1 center + 3 mid + 9 outer
    starts.append(_ring_config(1.0, 10, 0.5, 2, 1))                         # 1 center + 2 mid + 10 outer
    starts.append(_ring_config(1.0, 11, 0.4, 1, 1))                         # 1 center + 1 mid + 11 outer
    starts.append(_ring_config(1.0, 6, 0.65, 6, 1))                         # wider mid ring
    starts.append(_ring_config(1.0, 6, 0.35, 6, 1))                         # narrower mid ring
    starts.append(_ring_config(1.0, 7, 0.55, 5, 1))                         # wider mid ring
    starts.append(_ring_config(1.0, 7, 0.35, 5, 1))                         # narrower mid ring
    # k_hull / k_inner polygon splits
    for k_hull in [3, 4, 5, 6, 7, 8, 10]:
        k_inner = N - k_hull
        hull = _regular_polygon(k_hull, 1.0)
        if k_inner == 0:
            starts.append(hull)
        elif k_inner == 1:
            starts.append(np.vstack([hull, np.array([[0.0, 0.0]])]))
        else:
            inner = _regular_polygon(k_inner, 0.35, phase=np.pi / k_inner)
            starts.append(np.vstack([hull, inner]))

    # ---- Search: allocate a per-start budget ----
    best_points = None
    best_score = -float('inf')
    n_starts = len(starts)
    reserve_final = 15.0
    search_budget = time_limit - reserve_final
    per_start = search_budget / n_starts

    for idx, start_pts in enumerate(starts):
        # deadline for this start's local search
        deadline = start_time + min(start_time + search_budget - idx * per_start,
                                    time.time() + per_start + 2.0)
        if time.time() > start_time + search_budget:
            break
        try:
            cand, cand_s = _optimize_from(start_pts, deadline)
        except Exception:
            continue
        if cand_s > best_score:
            best_score = cand_s
            best_points = cand.copy()

    # ---- Final high-precision polish on the global best ----
    if best_points is not None and time.time() < start_time + time_limit - 8:
        area_ref = max(_hull_area(best_points), 1e-6)
        # High-beta soft pass
        for beta in [800.0, 1500.0]:
            if time.time() > start_time + time_limit - 6:
                break
            try:
                res = opt.minimize(
                    lambda x: _soft_objective(x, beta, area_ref),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 120},
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_score:
                    best_score = s; best_points = cand.copy()
            except Exception:
                pass

        # Long NM polish
        if time.time() < start_time + time_limit - 4:
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 800, 'adaptive': True, 'xatol': 1e-15, 'fatol': 1e-17},
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_score:
                    best_score = s; best_points = cand.copy()
            except Exception:
                pass

    # ---- Tiny-perturbation NM restarts from the best ----
    if best_points is not None and time.time() < start_time + time_limit - 3:
        for pert_seed in [42, 123, 456, 789]:
            if time.time() > start_time + time_limit - 2:
                break
            rng_pert = np.random.RandomState(pert_seed)
            perturbed = best_points + rng_pert.randn(N, 2) * 0.005
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-14, 'fatol': 1e-16},
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_score:
                    best_score = s; best_points = cand.copy()
            except Exception:
                pass

    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)

    return best_points

# EVOLVE-BLOCK-END
