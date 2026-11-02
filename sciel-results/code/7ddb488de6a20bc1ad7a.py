import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety

    # Precompute indices for all triples
    triples = list(combinations(range(N), 3))
    n_triples = len(triples)

    i_idx = np.array([t[0] for t in triples])
    j_idx = np.array([t[1] for t in triples])
    k_idx = np.array([t[2] for t in triples])

    def compute_score(points):
        """Compute min triangle area / convex hull area for given points."""
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0

        if area_hull < 1e-9:
            return 0.0, 0.0

        x = points[:, 0]
        y = points[:, 1]
        v1x = x[j_idx] - x[i_idx]
        v1y = y[j_idx] - y[i_idx]
        v2x = x[k_idx] - x[i_idx]
        v2y = y[k_idx] - y[i_idx]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        min_area = np.min(areas)
        score = min_area / area_hull
        return score, area_hull

    def _soft_min_score(points, beta=50.0):
        """
        Smooth surrogate: use log-sum-exp to approximate the minimum triangle area.
        Returns negative soft-min score for minimization.
        """
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6

        if area_hull < 1e-9:
            return 1e6

        x = points[:, 0]
        y = points[:, 1]
        v1x = x[j_idx] - x[i_idx]
        v1y = y[j_idx] - y[i_idx]
        v2x = x[k_idx] - x[i_idx]
        v2y = y[k_idx] - y[i_idx]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)

        # Soft-min via log-sum-exp
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        score = soft_min_area / area_hull
        return -score

    def _hard_score_neg(points_flat):
        """Negative hard score for final evaluation."""
        points = points_flat.reshape(-1, 2)
        score, _ = compute_score(points)
        return -score

    def _soft_objective(points_flat, beta=50.0):
        """Soft-min objective for gradient-based optimization."""
        points = points_flat.reshape(-1, 2)
        return _soft_min_score(points, beta)

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), start_angle=0.0):
        """Generate points of a regular n-gon."""
        angles = np.linspace(start_angle, start_angle + 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _make_initial_configurations():
        """Create several diverse initial configurations, emphasizing high hull counts."""
        configs = []

        # 13-gon (all on hull)
        configs.append(_regular_polygon(13, radius=1.0))

        # 13-gon with slight perturbation
        config = _regular_polygon(13, radius=1.0)
        config += np.random.normal(0, 0.02, config.shape)
        configs.append(config)

        # 12-gon + center
        hull_pts = _regular_polygon(12, radius=1.0)
        center = np.array([[0.0, 0.0]])
        configs.append(np.vstack([hull_pts, center]))

        # 11-gon + 2 interior (close to hull)
        hull_pts = _regular_polygon(11, radius=1.0)
        interior1 = np.array([[0.95 * np.cos(0.3), 0.95 * np.sin(0.3)]])
        interior2 = np.array([[0.95 * np.cos(0.9), 0.95 * np.sin(0.9)]])
        configs.append(np.vstack([hull_pts, interior1, interior2]))

        # 10-gon + 3 interior (on a small triangle)
        hull_pts = _regular_polygon(10, radius=1.0)
        interior = _regular_polygon(3, radius=0.4, start_angle=np.pi/6)
        configs.append(np.vstack([hull_pts, interior]))

        # 9-gon + 4 interior (square)
        hull_pts = _regular_polygon(9, radius=1.0)
        interior = _regular_polygon(4, radius=0.45, start_angle=np.pi/8)
        configs.append(np.vstack([hull_pts, interior]))

        # 8-gon + 5 interior (pentagon)
        hull_pts = _regular_polygon(8, radius=1.0)
        interior = _regular_polygon(5, radius=0.5, start_angle=np.pi/10)
        configs.append(np.vstack([hull_pts, interior]))

        # 7-gon + 6 interior (hexagon)
        hull_pts = _regular_polygon(7, radius=1.0)
        interior = _regular_polygon(6, radius=0.55, start_angle=np.pi/12)
        configs.append(np.vstack([hull_pts, interior]))

        # 6-gon + 7 interior (heptagon)
        hull_pts = _regular_polygon(6, radius=1.0)
        interior = _regular_polygon(7, radius=0.6, start_angle=np.pi/14)
        configs.append(np.vstack([hull_pts, interior]))

        # 5-gon + 8 interior
        hull_pts = _regular_polygon(5, radius=1.0)
        interior = _regular_polygon(8, radius=0.65, start_angle=np.pi/16)
        configs.append(np.vstack([hull_pts, interior]))

        # 4-gon + 9 interior
        hull_pts = _regular_polygon(4, radius=1.0, start_angle=np.pi/4)
        interior = _regular_polygon(9, radius=0.7, start_angle=np.pi/18)
        configs.append(np.vstack([hull_pts, interior]))

        # 3-gon + 10 interior
        hull_pts = _regular_polygon(3, radius=1.0, start_angle=np.pi/2)
        interior = _regular_polygon(10, radius=0.75, start_angle=np.pi/20)
        configs.append(np.vstack([hull_pts, interior]))

        return configs

    best_points = None
    best_score = -float('inf')

    configs = _make_initial_configurations()

    for i, config in enumerate(configs):
        if time.time() > start_time + time_limit - 20:
            break

        # Evaluate initial config
        s, _ = compute_score(config)
        if s > best_score:
            best_score = s
            best_points = config.copy()

        # Phase 1: L-BFGS-B with soft-min surrogate (multiple beta values for annealing)
        if time.time() > start_time + time_limit - 20:
            break

        current = config.copy()
        for beta in [15.0, 40.0, 80.0, 150.0]:
            if time.time() > start_time + time_limit - 20:
                break
            try:
                result = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 60, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                optimized = result.x.reshape(-1, 2)
                opt_score, _ = compute_score(optimized)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized.copy()
                current = optimized
            except Exception:
                pass

    # Phase 2: Nelder-Mead refinement with hard score on best so far
    if best_points is not None and time.time() < start_time + time_limit - 20:
        try:
            res = opt.minimize(
                _hard_score_neg,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 400, 'adaptive': True, 'xatol': 1e-11, 'fatol': 1e-13}
            )
            nm_points = res.x.reshape(-1, 2)
            nm_score, _ = compute_score(nm_points)
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_points.copy()
        except Exception:
            pass

    # Random restarts with focus on high-hull configurations
    if time.time() < start_time + time_limit - 20:
        np.random.seed(42)
        n_restarts = 12
        for r in range(n_restarts):
            if time.time() > start_time + time_limit - 20:
                break
            
            # Bias towards higher hull counts
            n_outer = np.random.choice([11, 12, 13, 10, 9, 8], p=[0.2, 0.2, 0.2, 0.15, 0.15, 0.1])
            n_inner = N - n_outer
            
            outer = _regular_polygon(n_outer, radius=np.random.uniform(0.9, 1.1),
                                     start_angle=np.random.uniform(0, 2*np.pi))
            if n_inner > 0:
                inner_radius = np.random.uniform(0.3, 0.6)
                inner = _regular_polygon(n_inner, radius=inner_radius,
                                         start_angle=np.random.uniform(0, 2*np.pi))
                config = np.vstack([outer, inner])
            else:
                config = outer
            
            # Small perturbation
            config = config + np.random.normal(0, 0.05, config.shape)

            s, _ = compute_score(config)
            if s > best_score:
                best_score = s
                best_points = config.copy()

            if time.time() > start_time + time_limit - 20:
                break

            current = config.copy()
            for beta in [25.0, 60.0, 120.0]:
                if time.time() > start_time + time_limit - 20:
                    break
                try:
                    result = opt.minimize(
                        lambda x, b=beta: _soft_objective(x, b),
                        current.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 50, 'ftol': 1e-13, 'gtol': 1e-11}
                    )
                    optimized = result.x.reshape(-1, 2)
                    opt_score, _ = compute_score(optimized)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = optimized.copy()
                    current = optimized
                except Exception:
                    pass

            # Quick NM polish
            if time.time() > start_time + time_limit - 20:
                break
            if best_points is not None:
                try:
                    res = opt.minimize(
                        _hard_score_neg,
                        best_points.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-11, 'fatol': 1e-13}
                    )
                    nm_points = res.x.reshape(-1, 2)
                    nm_score, _ = compute_score(nm_points)
                    if nm_score > best_score:
                        best_score = nm_score
                        best_points = nm_points.copy()
                except Exception:
                    pass

    # Final high-precision polish
    if best_points is not None and time.time() < start_time + time_limit - 10:
        try:
            result = opt.minimize(
                lambda x: _soft_objective(x, 250.0),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            polished = result.x.reshape(-1, 2)
            pol_score, _ = compute_score(polished)
            if pol_score > best_score:
                best_score = pol_score
                best_points = polished.copy()
        except Exception:
            pass

        if time.time() < start_time + time_limit - 5:
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14}
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

# EVOLVE-BLOCK-END
