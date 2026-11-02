import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # 5s margin for safety

    # Precompute indices for all triples
    triples = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
    idx_i, idx_j, idx_k = triples[:, 0], triples[:, 1], triples[:, 2]
    n_triples = len(triples)

    def compute_score(points):
        """Compute min triangle area / convex hull area for given points."""
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0
            
        if area_hull < 1e-9:
            return 0.0

        x = points[:, 0]
        y = points[:, 1]
        
        # Vectorized triangle area calculation
        v1x = x[idx_j] - x[idx_i]
        v1y = y[idx_j] - y[idx_i]
        v2x = x[idx_k] - x[idx_i]
        v2y = y[idx_k] - y[idx_i]
        
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        min_area = np.min(areas)
        
        return min_area / area_hull

    def get_areas(points):
        """Helper to get all triangle areas for soft-min."""
        x = points[:, 0]
        y = points[:, 1]
        
        v1x = x[idx_j] - x[idx_i]
        v1y = y[idx_j] - y[idx_i]
        v2x = x[idx_k] - x[idx_i]
        v2y = y[idx_k] - y[idx_i]
        
        cross = np.abs(v1x * v2y - v1y * v2x)
        return 0.5 * cross

    def _soft_min_score(points_flat, beta=50.0):
        """
        Smooth surrogate: use log-sum-exp to approximate the minimum triangle area.
        Returns negative soft-min score for minimization.
        """
        points = points_flat.reshape(-1, 2)
        
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6
            
        if area_hull < 1e-9:
            return 1e6

        areas = get_areas(points)
        
        # Soft-min via log-sum-exp: -log(sum(exp(-beta * a_i))) / beta approximates min(a_i)
        # For stability, subtract max
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        
        # Handle potential overflow/underflow
        exp_terms = np.exp(neg_beta_areas - max_val)
        log_sum_exp = max_val + np.log(np.sum(exp_terms))
        
        soft_min_area = -log_sum_exp / beta
        score = soft_min_area / area_hull
        return -score

    def _hard_score_neg(points_flat):
        """Negative hard score for final evaluation."""
        points = points_flat.reshape(-1, 2)
        score = compute_score(points)
        return -score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        """Generate points of a regular n-gon."""
        if n < 3:
            # For n=1 or 2, just return points on a line or single point
            if n == 1:
                return np.array([center])
            elif n == 2:
                return np.array([[center[0] - radius, center[1]], [center[0] + radius, center[1]]])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _make_initial_configurations():
        """Create several diverse initial configurations."""
        configs = []
        
        # 1. Regular polygon (all on hull)
        configs.append(_regular_polygon(13, radius=1.0))
        
        # 2. 12 on hull, 1 center
        hull_pts = _regular_polygon(12, radius=1.0)
        center = np.array([[0.0, 0.0]])
        configs.append(np.vstack([hull_pts, center]))
        
        # 3. 11 on hull, 2 interior
        hull_pts = _regular_polygon(11, radius=1.0)
        interior1 = np.array([[0.3, 0.0]])
        interior2 = np.array([[-0.15, 0.26]])
        configs.append(np.vstack([hull_pts, interior1, interior2]))
        
        # 4. 10 on hull, 3 interior (triangle)
        hull_pts = _regular_polygon(10, radius=1.0)
        interior = _regular_polygon(3, radius=0.3)
        configs.append(np.vstack([hull_pts, interior]))
        
        # 5. 9 on hull, 4 interior (square)
        hull_pts = _regular_polygon(9, radius=1.0)
        interior = _regular_polygon(4, radius=0.35)
        configs.append(np.vstack([hull_pts, interior]))
        
        # 6. 7 on hull, 6 interior (hexagon)
        outer = _regular_polygon(7, radius=1.0)
        inner = _regular_polygon(6, radius=0.5)
        configs.append(np.vstack([outer, inner]))
        
        # 7. 8 on hull, 5 interior (pentagon)
        outer = _regular_polygon(8, radius=1.0)
        inner = _regular_polygon(5, radius=0.45)
        configs.append(np.vstack([outer, inner]))
        
        # 8. 9 on hull, 4 interior (square)
        outer = _regular_polygon(9, radius=1.0)
        inner = _regular_polygon(4, radius=0.4)
        configs.append(np.vstack([outer, inner]))
        
        # 9. 10 on hull, 3 interior (triangle)
        outer = _regular_polygon(10, radius=1.0)
        inner = _regular_polygon(3, radius=0.3)
        configs.append(np.vstack([outer, inner]))
        
        # 10. 11 on hull, 2 interior
        outer = _regular_polygon(11, radius=1.0)
        inner = np.array([[0.2, 0.2], [-0.2, -0.2]])
        configs.append(np.vstack([outer, inner]))
        
        # 11. Randomized perturbation of regular polygon
        base = _regular_polygon(13, radius=1.0)
        rng = np.random.default_rng(42)
        perturbed = base + rng.normal(0, 0.1, base.shape)
        configs.append(perturbed)
        
        # 12. Random points in unit circle
        rng2 = np.random.default_rng(123)
        angles = rng2.uniform(0, 2*np.pi, 13)
        radii = rng2.uniform(0, 1, 13)
        random_pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        configs.append(random_pts)

        return configs

    best_points = None
    best_score = -float('inf')
    
    configs = _make_initial_configurations()
    
    for i, config in enumerate(configs):
        if time.time() > start_time + time_limit - 15:
            break
        
        # Evaluate initial config
        s = compute_score(config)
        if s > best_score:
            best_score = s
            best_points = config.copy()
        
        # Phase 1: L-BFGS-B with soft-min surrogate (multiple beta values for annealing)
        if time.time() > start_time + time_limit - 15:
            break
        
        current = config.copy()
        # Annealing schedule: start loose, get tight
        betas = [10.0, 30.0, 100.0]
        
        for beta in betas:
            if time.time() > start_time + time_limit - 15:
                break
            try:
                result = opt.minimize(
                    lambda x: _soft_min_score(x, beta),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                optimized = result.x.reshape(-1, 2)
                opt_score = compute_score(optimized)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized.copy()
                
                # Use the result as the starting point for the next tighter beta
                current = optimized
            except Exception:
                pass
        
        # Phase 2: Nelder-Mead refinement with hard score
        if time.time() > start_time + time_limit - 15:
            break
        
        if best_points is not None:
            try:
                # Start NM from the best found so far, or the local best from L-BFGS
                start_nm = current.copy()
                res = opt.minimize(
                    _hard_score_neg,
                    start_nm.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                nm_points = res.x.reshape(-1, 2)
                nm_score = compute_score(nm_points)
                if nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_points.copy()
            except Exception:
                pass

    # If we have time, do a few random restarts with the best configuration as a seed
    if time.time() < start_time + time_limit - 10 and best_points is not None:
        rng = np.random.default_rng(999)
        n_restarts = 5
        for _ in range(n_restarts):
            if time.time() > start_time + time_limit - 10:
                break
            # Perturb best points
            noise_scale = 0.05
            perturbed = best_points + rng.normal(0, noise_scale, best_points.shape)
            
            # Quick L-BFGS with high beta
            try:
                result = opt.minimize(
                    lambda x: _soft_min_score(x, 100.0),
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                opt_pts = result.x.reshape(-1, 2)
                s = compute_score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_points = opt_pts.copy()
            except Exception:
                pass

    if best_points is None:
        # Fallback
        best_points = _regular_polygon(13, radius=1.0)

    return best_points
