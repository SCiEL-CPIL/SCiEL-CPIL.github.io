import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety
    
    # Precompute indices for all triples
    triples = [(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)]
    n_triples = len(triples)
    
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
        
        # Vectorized triangle area calculation
        # For each triple (i, j, k), area = 0.5 * |det(p_j - p_i, p_k - p_i)|
        # We can compute this for all triples at once using broadcasting, 
        # but 286 triples is small enough that a loop or simple vectorization is fine.
        # Let's use a vectorized approach for speed.
        
        # Expand triples to arrays
        i_idx = np.array([t[0] for t in triples])
        j_idx = np.array([t[1] for t in triples])
        k_idx = np.array([t[2] for t in triples])
        
        xi = x[i_idx]
        yi = y[i_idx]
        xj = x[j_idx]
        yj = y[j_idx]
        xk = x[k_idx]
        yk = y[k_idx]
        
        # Cross product magnitude
        # det = (xj - xi)*(yk - yi) - (yj - yi)*(xk - xi)
        #     = xj*yk - xj*yi - xi*yk + xi*yi - yj*xk + yj*xi + yi*xk - yi*xi
        #     = xj*yk - yj*xk + xi*(yj - yk) + yi*(xk - xj)
        
        cross = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        score = min_area / area_hull
        return score, area_hull

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

        x = points[:, 0]
        y = points[:, 1]
        
        # Compute all triangle areas
        i_idx = np.array([t[0] for t in triples])
        j_idx = np.array([t[1] for t in triples])
        k_idx = np.array([t[2] for t in triples])
        
        xi = x[i_idx]
        yi = y[i_idx]
        xj = x[j_idx]
        yj = y[j_idx]
        xk = x[k_idx]
        yk = y[k_idx]
        
        cross = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
        areas = 0.5 * np.abs(cross)

        # Soft-min via log-sum-exp: -log(sum(exp(-beta * a_i))) / beta approximates min(a_i)
        # For stability, subtract max
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        # If max_val is very large negative, exp(neg_beta_areas - max_val) is safe
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta

        score = soft_min_area / area_hull
        return -score

    def _hard_score_neg(points_flat):
        """Negative hard score for final evaluation."""
        points = points_flat.reshape(-1, 2)
        score, _ = compute_score(points)
        return -score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        """Generate points of a regular n-gon."""
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _make_initial_configurations():
        """Create several diverse initial configurations."""
        configs = []
        
        # 1. Regular 13-gon
        configs.append(_regular_polygon(13, radius=1.0))
        
        # 2. 12-gon + Center
        hull_pts = _regular_polygon(12, radius=1.0)
        center = np.array([[0.0, 0.0]])
        configs.append(np.vstack([hull_pts, center]))
        
        # 3. 11-gon + 2 Interior (symmetric)
        hull_pts = _regular_polygon(11, radius=1.0)
        # Place 2 points symmetrically
        ang1 = 2 * np.pi * 0.5 / 11
        ang2 = 2 * np.pi * 1.5 / 11
        r_inner = 0.4
        p1 = np.array([r_inner * np.cos(ang1), r_inner * np.sin(ang1)])
        p2 = np.array([r_inner * np.cos(ang2), r_inner * np.sin(ang2)])
        configs.append(np.vstack([hull_pts, p1, p2]))
        
        # 4. 10-gon + 3 Interior (triangle)
        hull_pts = _regular_polygon(10, radius=1.0)
        inner = _regular_polygon(3, radius=0.35)
        configs.append(np.vstack([hull_pts, inner]))
        
        # 5. 9-gon + 4 Interior (square)
        hull_pts = _regular_polygon(9, radius=1.0)
        inner = _regular_polygon(4, radius=0.4)
        configs.append(np.vstack([hull_pts, inner]))
        
        # 6. 8-gon + 5 Interior (pentagon)
        hull_pts = _regular_polygon(8, radius=1.0)
        inner = _regular_polygon(5, radius=0.45)
        configs.append(np.vstack([hull_pts, inner]))
        
        # 7. 7-gon + 6 Interior (hexagon)
        hull_pts = _regular_polygon(7, radius=1.0)
        inner = _regular_polygon(6, radius=0.5)
        configs.append(np.vstack([hull_pts, inner]))
        
        # 8. 6-gon + 7 Interior (heptagon)
        hull_pts = _regular_polygon(6, radius=1.0)
        inner = _regular_polygon(7, radius=0.55)
        configs.append(np.vstack([hull_pts, inner]))
        
        return configs

    best_points = None
    best_score = -float('inf')
    
    configs = _make_initial_configurations()
    
    for i, config in enumerate(configs):
        if time.time() > start_time + time_limit - 10:
            break
        
        # Evaluate initial config
        s, _ = compute_score(config)
        if s > best_score:
            best_score = s
            best_points = config.copy()
        
        # Phase 1: L-BFGS-B with soft-min surrogate (multiple beta values for annealing)
        if time.time() > start_time + time_limit - 10:
            break
        
        current = config.copy()
        # Use increasing beta to sharpen the minimum approximation
        for beta in [20.0, 50.0, 100.0]:
            if time.time() > start_time + time_limit - 10:
                break
            try:
                result = opt.minimize(
                    lambda x: _soft_min_score(x, beta),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                optimized = result.x.reshape(-1, 2)
                opt_score, _ = compute_score(optimized)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized.copy()
                # Keep the soft-min optimized point for next beta
                current = optimized
            except Exception:
                pass
        
        # Phase 2: Nelder-Mead refinement with hard score on the best found so far
        if time.time() > start_time + time_limit - 10:
            break
        
        if best_points is not None:
            try:
                # Use a copy to avoid modifying best_points if NM fails or worsens
                nm_start = best_points.copy()
                res = opt.minimize(
                    _hard_score_neg,
                    nm_start.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                nm_points = res.x.reshape(-1, 2)
                nm_score, _ = compute_score(nm_points)
                if nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_points.copy()
            except Exception:
                pass

    # If time remains, try a few random restarts or perturbations of the best point
    if time.time() < start_time + time_limit - 5 and best_points is not None:
        np.random.seed(42)
        # Try perturbing the best point slightly and re-optimizing with hard NM
        for _ in range(3):
            if time.time() > start_time + time_limit - 5:
                break
            perturbed = best_points + np.random.normal(0, 0.05, best_points.shape)
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                p_points = res.x.reshape(-1, 2)
                p_score, _ = compute_score(p_points)
                if p_score > best_score:
                    best_score = p_score
                    best_points = p_points.copy()
            except Exception:
                pass

    if best_points is None:
        # Fallback: Regular polygon
        best_points = _regular_polygon(13, radius=1.0)
        
    return best_points

# EVOLVE-BLOCK-END
