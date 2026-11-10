import numpy as np
import time
import scipy.spatial
import scipy.optimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the smallest triangle area
    to the convex hull area.
    
    Strategy:
    1. Parameterize points in 2D Cartesian coordinates.
    2. Use a smooth surrogate objective (soft-min of triangle areas) to allow gradient-based optimization.
    3. Multi-start L-BFGS-B optimization with increasing beta (sharpness).
    4. Final polish with Nelder-Mead on the hard objective.
    """
    n = 13
    start_time = time.time()
    time_limit = 580  # Safety margin before 600s limit
    
    # Precompute indices for all 286 triangles
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    tri_i = i_idx[mask]
    tri_j = j_idx[mask]
    tri_k = k_idx[mask]
    
    def calculate_metric(points, beta):
        """
        Calculate the smooth surrogate metric.
        beta controls the sharpness of the min approximation.
        """
        try:
            hull = scipy.spatial.ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-9:
                return 0.0
        except Exception:
            return 0.0
            
        p_i = points[tri_i]
        p_j = points[tri_j]
        p_k = points[tri_k]
        
        # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        if np.min(areas) < 1e-9:
            return 0.0
            
        # Stable soft-min approximation
        m = np.min(areas)
        shifted = areas - m
        exp_terms = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0:
            soft_min = m
        else:
            soft_min = m - (1.0 / beta) * np.log(sum_exp)
            
        return soft_min / A_hull

    def neg_objective_smooth(coords, beta):
        """Objective for L-BFGS-B: negative of soft metric."""
        points = coords.reshape(n, 2)
        val = calculate_metric(points, beta)
        return -val

    # Initial Guess: Regular 13-gon
    theta0 = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0 = np.column_stack((np.cos(theta0), np.sin(theta0))).flatten()
    
    best_score = -1.0
    best_coords = x0.copy()
    
    # Multi-start with different initial perturbations
    rng = np.random.default_rng(seed=42)
    
    # Generate multiple starting points
    starts = [x0.copy()]
    
    # Add perturbed starts
    for _ in range(3):
        perturbation = rng.normal(0, 0.2, size=2*n)
        perturbed = x0 + perturbation
        starts.append(perturbed)
    
    # Simulated Annealing / Gradient Descent Loop with increasing beta
    betas = [10, 50, 100, 200, 500, 1000, 2000]
    
    current_coords = None
    
    for beta in betas:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:
            break
            
        for start in starts:
            if current_coords is not None:
                # Use the previous best result as the start for the next higher beta
                x0_iter = current_coords
            else:
                x0_iter = start
                
            try:
                res = scipy.optimize.minimize(
                    neg_objective_smooth, 
                    x0_iter, 
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                # Evaluate the true metric (hard min) for the result
                cand_points = res.x.reshape(n, 2)
                hard_val = calculate_metric(cand_points, beta=10000) # Use large beta for "hard" eval
                
                if hard_val > best_score:
                    best_score = hard_val
                    best_coords = res.x.copy()
                    current_coords = res.x.copy()
                    
            except Exception:
                pass
                
        # After finishing a beta level, if we have a good current_coords, 
        # we can use it as the start for the next beta level for all "starts" 
        # effectively merging them into a single trajectory.
    
    # Final Polish with Nelder-Mead on the hard objective
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30:
        try:
            def neg_objective_hard(coords):
                points = coords.reshape(n, 2)
                return -calculate_metric(points, beta=10000)
                
            res_final = scipy.optimize.minimize(
                neg_objective_hard, 
                best_coords, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            final_hard_val = -res_final.fun
            if final_hard_val > best_score:
                best_score = final_hard_val
                best_coords = res_final.x.copy()
                
        except Exception:
            pass

    final_points = best_coords.reshape(n, 2)
    
    # Ensure validity
    if not np.all(np.isfinite(final_points)):
        # Fallback to regular polygon
        final_points = np.column_stack((np.cos(theta0), np.sin(theta0)))

    return final_points

# EVOLVE-BLOCK-END
