import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
TIME_LIMIT = 145.0  # Leave 5s margin before 150s kill

def construct():
    start_time = time.time()
    time_limit = TIME_LIMIT

    # Precompute triple index arrays for vectorized area computation
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_triples = len(tri_i)

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
        # Vectorized cross products for all 286 triples
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        min_area = float(np.min(areas))

        score = min_area / area_hull
        return score, area_hull

    def _soft_min_score(points, beta=100.0):
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

        # Vectorized triangle areas
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = v1x * v2y - v1y * v2x
        areas = 0.5 * np.abs(cross)

        # Soft-min via log-sum-exp (stable)
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta

        score = soft_min_area / area_hull
        return -score

    def _soft_min_grad(points, beta=100.0):
        """
        Compute gradient of the soft-min score w.r.t. point coordinates.
        Returns (grad_x, grad_y) arrays of shape (N,).
        """
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return None, None

        if area_hull < 1e-9:
            return None, None

        x = points[:, 0]
        y = points[:, 1]

        # Triangle areas and their derivatives
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = v1x * v2y - v1y * v2x
        areas = 0.5 * np.abs(cross)

        # Sign of cross product for differentiability
        sign_cross = np.sign(cross)
        
        # Weights for soft-min: w_t = exp(-beta * area_t) / sum(exp(-beta * area_t))
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        exp_vals = np.exp(neg_beta_areas - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp  # w_t

        # d(soft_min_area)/d(area_t) = -w_t
        # d(area_t)/d(point) = 0.5 * sign(cross_t) * d(cross_t)/d(point)
        # d(score)/d(point) = d(soft_min_area/area_hull)/d(point) = (d(soft_min_area)/d(point)) / area_hull
        # Since we want to maximize score, we minimize -score, so gradient of -score = -gradient of score

        # Compute d(soft_min_area)/dx and d(soft_min_area)/dy
        d_soft_min_dx = np.zeros(N)
        d_soft_min_dy = np.zeros(N)

        # For each triangle, accumulate contributions
        # coeff = -weights * 0.5 * sign_cross
        coeff = -0.5 * weights * sign_cross 

        # d(cross)/dx_i = y_j - y_k
        d_soft_min_dx[tri_i] += coeff * (y[tri_j] - y[tri_k])
        # d(cross)/dx_j = y_k - y_i
        d_soft_min_dx[tri_j] += coeff * (y[tri_k] - y[tri_i])
        # d(cross)/dx_k = -(y_j - y_i)
        d_soft_min_dx[tri_k] += coeff * (-(y[tri_j] - y[tri_i]))

        # d(cross)/dy_i = x_j - x_k
        d_soft_min_dy[tri_i] += coeff * (x[tri_j] - x[tri_k])
        # d(cross)/dy_j = -(x_k - x_i)
        d_soft_min_dy[tri_j] += coeff * (-(x[tri_k] - x[tri_i]))
        # d(cross)/dy_k = x_j - x_i
        d_soft_min_dy[tri_k] += coeff * (x[tri_j] - x[tri_i])

        # d(score)/dx = d(soft_min_area)/dx / area_hull
        # d(-score)/dx = -d(score)/dx
        grad_x = -d_soft_min_dx / area_hull
        grad_y = -d_soft_min_dy / area_hull

        return grad_x, grad_y

    def _direct_bottleneck_optimize(points, max_iter=50, time_budget=10.0):
        """
        Direct local search targeting the worst-performing triangles.
        Iteratively identifies triangles with minimum area and moves their vertices
        to increase that area, ensuring the global minimum does not decrease.
        """
        start_t = time.time()
        current = points.copy()
        best = points.copy()
        best_s, _ = compute_score(best)
        
        # Precompute area vectors for speed
        def get_areas(pts):
            x = pts[:, 0]
            y = pts[:, 1]
            v1x = x[tri_j] - x[tri_i]
            v1y = y[tri_j] - y[tri_i]
            v2x = x[tri_k] - x[tri_i]
            v2y = y[tri_k] - y[tri_i]
            return 0.5 * np.abs(v1x * v2y - v1y * v2x)

        for it in range(max_iter):
            if time.time() - start_t > time_budget:
                break
            
            areas = get_areas(current)
            min_area = np.min(areas)
            
            # Find all triangles that are within a small epsilon of the minimum
            # This helps in balancing the "active set"
            tol = 1e-5
            active_mask = areas <= min_area + tol
            active_indices = np.where(active_mask)[0]
            
            if len(active_indices) == 0:
                break
            
            # Compute gradient of the sum of active triangle areas
            # We want to move points to increase these areas.
            # d(area)/d(p) for a triangle (i,j,k)
            x = current[:, 0]
            y = current[:, 1]
            
            grad_x = np.zeros(N)
            grad_y = np.zeros(N)
            
            # Only compute for active triangles to focus effort
            # Note: cross product sign matters for gradient direction
            v1x_all = x[tri_j] - x[tri_i]
            v1y_all = y[tri_j] - y[tri_i]
            v2x_all = x[tri_k] - x[tri_i]
            v2y_all = y[tri_k] - y[tri_i]
            cross_all = v1x_all * v2y_all - v1y_all * v2x_all
            sign_all = np.sign(cross_all)
            
            # Weight active triangles equally for this step
            weights = np.zeros(n_triples)
            weights[active_indices] = 1.0 / len(active_indices)
            
            coeff_x = 0.5 * weights * sign_all
            coeff_y = 0.5 * weights * sign_all
            
            # Accumulate gradients for active triangles only
            # To avoid full vector operations, we can just do it for all and mask, 
            # or loop. With 286 triples, vectorized is fine.
            d_dx_i = coeff_x * (y[tri_j] - y[tri_k])
            d_dx_j = coeff_x * (y[tri_k] - y[tri_i])
            d_dx_k = coeff_x * (-(y[tri_j] - y[tri_i]))
            
            d_dy_i = coeff_y * (x[tri_j] - x[tri_k])
            d_dy_j = coeff_y * (-(x[tri_k] - x[tri_i]))
            d_dy_k = coeff_y * (x[tri_j] - x[tri_i])
            
            grad_x[tri_i] += d_dx_i
            grad_x[tri_j] += d_dx_j
            grad_x[tri_k] += d_dx_k
            
            grad_y[tri_i] += d_dy_i
            grad_y[tri_j] += d_dy_j
            grad_y[tri_k] += d_dy_k
            
            # Normalize gradient
            norm_g = np.linalg.norm(np.concatenate([grad_x, grad_y]))
            if norm_g < 1e-8:
                break
            dir_x = grad_x / norm_g
            dir_y = grad_y / norm_g
            
            # Line search: find step size that increases min area
            step = 0.1
            improved = False
            for _ in range(20):
                trial_x = current[:, 0] + step * dir_x
                trial_y = current[:, 1] + step * dir_y
                trial_pts = np.column_stack([trial_x, trial_y])
                
                trial_areas = get_areas(trial_pts)
                trial_min = np.min(trial_areas)
                
                if trial_min > min_area * (1 + 1e-7):
                    current = trial_pts
                    improved = True
                    
                    # Check overall score
                    s, _ = compute_score(current)
                    if s > best_s:
                        best_s = s
                        best = current.copy()
                    break
                step *= 0.5
                
            if not improved:
                break

        return best, best_s

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        """Generate points of a regular n-gon."""
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _ring_config(r1, r2, r3, n1, n2, n3, phase1=0.0, phase2=0.0, phase3=0.0):
        """
        Symmetric ring configuration with 3 rings.
        """
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
        """Run multi-stage optimization from a starting configuration."""
        best = start_pts.copy()
        best_s, _ = compute_score(best)
        current = start_pts.copy()

        # Stage 1: L-BFGS-B with annealed soft-min beta
        # Use lower betas first to escape local minima, higher betas to sharpen
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() - start_time > time_budget:
                break
            try:
                res = opt.minimize(
                    lambda x: _soft_min_score(x.reshape(-1, 2), beta),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
                current = cand
            except Exception:
                pass

        # Stage 2: Direct Bottleneck Optimization
        # This is the key improvement suggested by the findings
        if time.time() - start_time < time_budget - 10:
            try:
                cand, cand_s = _direct_bottleneck_optimize(best, max_iter=100, time_budget=5.0)
                if cand_s > best_s:
                    best_s = cand_s
                    best = cand.copy()
            except Exception:
                pass
            
            # Repeat bottleneck optimization with slightly perturbed start if time allows
            if time.time() - start_time < time_budget - 5:
                try:
                    noise = np.random.normal(0, 0.01, best.shape)
                    cand2, cand_s2 = _direct_bottleneck_optimize(best + noise, max_iter=50, time_budget=3.0)
                    if cand_s2 > best_s:
                        best_s = cand_s2
                        best = cand2.copy()
                except Exception:
                    pass

        # Stage 3: Powell method for robust derivative-free refinement
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

        # Stage 4: Nelder-Mead polish on hard objective
        if time.time() - start_time < time_budget - 2:
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

    # --- Phase 1: Symmetric ring initializations ---
    ring_starts = [
        # 1 center + 12 outer
        _ring_config(r1=1.0, r2=0.0, r3=0.0, n1=12, n2=0, n3=1),
        # 1 center + 6 mid + 6 outer
        _ring_config(r1=1.0, r2=0.5, r3=0.0, n1=6, n2=6, n3=1),
        # 1 center + 5 mid + 7 outer
        _ring_config(r1=1.0, r2=0.45, r3=0.0, n1=7, n2=5, n3=1),
        # 1 center + 4 mid + 8 outer
        _ring_config(r1=1.0, r2=0.4, r3=0.0, n1=8, n2=4, n3=1),
        # 1 center + 12 outer with phase offset
        _ring_config(r1=1.0, r2=0.0, r3=0.0, n1=12, n2=0, n3=1, phase1=np.pi/12),
        # 2 center + 11 outer
        _ring_config(r1=1.0, r2=0.3, r3=0.0, n1=11, n2=2, n3=0, phase2=np.pi/2),
        # 3 center + 10 outer
        _ring_config(r1=1.0, r2=0.4, r3=0.0, n1=10, n2=3, n3=0, phase2=np.pi/3),
        # 1 center + 3 mid + 9 outer
        _ring_config(r1=1.0, r2=0.6, r3=0.0, n1=9, n2=3, n3=1),
        # 1 center + 2 mid + 10 outer
        _ring_config(r1=1.0, r2=0.7, r3=0.0, n1=10, n2=2, n3=1),
        # 1 center + 4 mid + 8 outer with different phase
        _ring_config(r1=1.0, r2=0.5, r3=0.0, n1=8, n2=4, n3=1, phase2=np.pi/4),
        # Additional diverse configurations
        _ring_config(r1=1.0, r2=0.35, r3=0.0, n1=9, n2=3, n3=1, phase2=np.pi/6),
        _ring_config(r1=1.0, r2=0.55, r3=0.0, n1=7, n2=5, n3=1, phase2=np.pi/3),
        _ring_config(r1=1.0, r2=0.45, r3=0.0, n1=10, n2=2, n3=1, phase2=np.pi/5),
        _ring_config(r1=1.0, r2=0.65, r3=0.0, n1=8, n2=4, n3=1, phase2=np.pi/7),
        _ring_config(r1=1.0, r2=0.5, r3=0.0, n1=11, n2=1, n3=1, phase2=np.pi/2),
    ]

    n_starts = len(ring_starts)
    search_budget = time_limit - 15.0  # Reserve time for final polish
    per_start = search_budget / n_starts

    for idx, start_pts in enumerate(ring_starts):
        if time.time() > start_time + time_limit - 15:
            break
        
        # Give more time to the first few promising symmetric configs
        if idx < 5:
            alloc = per_start * 1.5
        else:
            alloc = per_start * 0.8
            
        remaining = time_limit - 15 - (time.time() - start_time)
        alloc = min(alloc, remaining)
        
        if alloc < 3.0:
            break

        cand, cand_s = _optimize_from(start_pts, time.time() - start_time + alloc)
        if cand_s > best_score:
            best_score = cand_s
            best_points = cand.copy()

    # Final high-precision polish
    if best_points is not None and time.time() < start_time + time_limit - 10:
        # Stage 1: High-beta soft-min L-BFGS-B
        try:
            result = opt.minimize(
                lambda x: _soft_min_score(x.reshape(-1, 2), 500.0),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 80}
            )
            polished = result.x.reshape(-1, 2)
            pol_score, _ = compute_score(polished)
            if pol_score > best_score:
                best_score = pol_score
                best_points = polished.copy()
        except Exception:
            pass
        
        # Stage 2: Direct Bottleneck Optimization on the best result
        if time.time() < start_time + time_limit - 5:
            try:
                cand, cand_s = _direct_bottleneck_optimize(best_points, max_iter=200, time_budget=5.0)
                if cand_s > best_score:
                    best_score = cand_s
                    best_points = cand.copy()
            except Exception:
                pass

        # Stage 3: Final NM polish
        if time.time() < start_time + time_limit - 2:
            try:
                res = opt.minimize(
                    lambda x: -compute_score(x.reshape(-1, 2))[0],
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'adaptive': True, 'xatol': 1e-13, 'fatol': 1e-15}
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
