import numpy as np
from scipy.optimize import differential_evolution
import time

def _vertex_score(polygon, i):
    """Calculate the score for vertex i of the polygon."""
    v = polygon[i]
    # Compute distances to all other vertices
    diffs = polygon - v
    diffs = np.delete(diffs, i, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    dists = np.sort(dists)
    
    # Check minimum distance
    if dists[0] < 0.05:
        return 0.0
        
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = np.diff(dists)
    min_gap = np.min(gaps)
    k = np.argmin(gaps)
    
    D = (dists[k] + dists[k+1]) / 2.0
    if D < 0.1:
        return 0.0
        
    # Select 4 distances
    if k == 0:
        ds = dists[0:4]
    elif k == 17:
        ds = dists[15:19]
    else:
        ds = dists[k-1:k+3]
        
    # Calculate raw score
    ratios = np.maximum(ds / D, D / ds)
    raw = np.mean(ratios)
    
    if raw == 0:
        return 1.0
    return min(1.0, 1.0 / raw)

def _total_score(polygon):
    """Calculate the minimum vertex score over all vertices."""
    min_score = 1.0
    for i in range(polygon.shape[0]):
        s = _vertex_score(polygon, i)
        if s < min_score:
            min_score = s
            if min_score == 0.0:
                return 0.0
    return min_score

def _is_convex(polygon):
    """Check if polygon is strictly convex (counter-clockwise)."""
    n = polygon.shape[0]
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross <= 1e-9:
            return False
    return True

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Regular polygon as baseline
    R_reg = np.ones(n)
    x_reg = R_reg * np.cos(angles)
    y_reg = R_reg * np.sin(angles)
    polygon_reg = np.stack([x_reg, y_reg], axis=1)
    
    # Normalize baseline
    centroid = np.mean(polygon_reg, axis=0)
    polygon_reg = polygon_reg - centroid
    mean_dist = np.mean(np.linalg.norm(polygon_reg, axis=1))
    if mean_dist > 0.01:
        polygon_reg = polygon_reg / mean_dist
    else:
        return polygon_reg
        
    score_reg = _total_score(polygon_reg)
    
    # Try to optimize using Fourier perturbation + differential evolution
    start_time = time.time()
    time_limit = 600.0
    current_best = score_reg
    best_polygon = polygon_reg.copy()
    
    # Use low-frequency Fourier series for radial perturbation
    # R_i = 1 + sum_{m=1}^{M} (a_m * cos(m*theta_i) + b_m * sin(m*theta_i))
    # Limit to low frequencies (m=1,2,3) to maintain convexity
    M = 3  # Number of Fourier harmonics
    n_params = 2 * M  # a_1, b_1, a_2, b_2, a_3, b_3
    
    def fourier_to_R(params):
        """Convert Fourier coefficients to radial distances."""
        a = params[:M]
        b = params[M:]
        R = np.ones(n)
        for m in range(1, M + 1):
            R += a[m-1] * np.cos(m * angles) + b[m-1] * np.sin(m * angles)
        return R
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 30:
            return 1e9
            
        R = fourier_to_R(params)
        if np.any(R <= 0.3):  # Ensure positive radii
            return 1e9
            
        x = R * np.cos(angles)
        y = R * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
        
        # Check convexity
        if not _is_convex(polygon):
            return 1e9 + (time.time() - start_time)
            
        s = _total_score(polygon)
        return -s
        
    # Bounds for Fourier coefficients: keep perturbation small to maintain convexity
    # For a regular polygon, small perturbations preserve convexity
    # a_m, b_m should be small: |a_m|, |b_m| < 0.1 for safety
    bounds = [(-0.08, 0.08)] * n_params
    
    # Run differential evolution
    try:
        res = differential_evolution(
            objective, 
            bounds=bounds,
            seed=42,
            maxiter=500,
            tol=1e-6,
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,
            callback=None
        )
        
        if res.fun < -current_best:
            R_opt = fourier_to_R(res.x)
            x_opt = R_opt * np.cos(angles)
            y_opt = R_opt * np.sin(angles)
            polygon_opt = np.stack([x_opt, y_opt], axis=1)
            
            # Verify convexity again
            if _is_convex(polygon_opt):
                # Normalize
                centroid_opt = np.mean(polygon_opt, axis=0)
                polygon_opt = polygon_opt - centroid_opt
                mean_dist_opt = np.mean(np.linalg.norm(polygon_opt, axis=1))
                if mean_dist_opt > 0.01:
                    polygon_opt = polygon_opt / mean_dist_opt
                    score_opt = _total_score(polygon_opt)
                    if score_opt > current_best:
                        best_polygon = polygon_opt
                        current_best = score_opt
    except Exception:
        pass
    
    # If time remains, try local refinement with Nelder-Mead on the best found
    elapsed = time.time() - start_time
    if elapsed < time_limit - 60 and current_best > score_reg:
        try:
            # Extract R from best_polygon for local optimization
            # Reconstruct angles and get current R
            x_best = best_polygon[:, 0]
            y_best = best_polygon[:, 1]
            R_current = np.sqrt(x_best**2 + y_best**2)
            
            def objective_local(params):
                elapsed_local = time.time() - start_time
                if elapsed_local > time_limit - 10:
                    return 1e9
                    
                R = params
                if np.any(R <= 0.3):
                    return 1e9
                    
                x = R * np.cos(angles)
                y = R * np.sin(angles)
                polygon = np.stack([x, y], axis=1)
                
                if not _is_convex(polygon):
                    return 1e9
                    
                s = _total_score(polygon)
                return -s
                
            from scipy.optimize import minimize
            res_local = minimize(objective_local, R_current, method='Nelder-Mead',
                                 options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7})
            
            if res_local.fun < -current_best:
                R_opt = res_local.x
                x_opt = R_opt * np.cos(angles)
                y_opt = R_opt * np.sin(angles)
                polygon_opt = np.stack([x_opt, y_opt], axis=1)
                
                if _is_convex(polygon_opt):
                    centroid_opt = np.mean(polygon_opt, axis=0)
                    polygon_opt = polygon_opt - centroid_opt
                    mean_dist_opt = np.mean(np.linalg.norm(polygon_opt, axis=1))
                    if mean_dist_opt > 0.01:
                        polygon_opt = polygon_opt / mean_dist_opt
                        score_opt = _total_score(polygon_opt)
                        if score_opt > current_best:
                            best_polygon = polygon_opt
                            current_best = score_opt
        except Exception:
            pass
        
    return best_polygon

# EVOLVE-BLOCK-END
