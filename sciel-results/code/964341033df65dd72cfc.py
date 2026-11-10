import numpy as np
import time
import scipy.spatial
import scipy.optimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the smallest triangle area
    to the convex hull area.
    
    Strategy:
    1. Optimize 26 coordinates directly (full 2D freedom).
    2. Use a soft-min surrogate for the minimum triangle area with increasing sharpness (beta).
    3. Multi-start L-BFGS-B optimization.
    4. Final polish with Nelder-Mead on the hard objective.
    """
    n = 13
    start_time = time.time()
    time_limit = 580  # Safety margin before 600s limit
    
    # Precompute indices for triangles
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    tri_i = i_idx[mask]
    tri_j = j_idx[mask]
    tri_k = k_idx[mask]
    num_tris = len(tri_i) # 286

    def get_areas(points):
        """Compute areas of all 286 triangles."""
        p_i = points[tri_i]
        p_j = points[tri_j]
        p_k = points[tri_k]
        
        # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        return areas

    def get_hull_area(points):
        """Compute area of the convex hull."""
        try:
            hull = scipy.spatial.ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def soft_min(areas, beta):
        """
        Stable soft-min approximation using log-sum-exp.
        soft_min = -1/beta * log(sum(exp(-beta * areas)))
        """
        if len(areas) == 0:
            return 0.0
        m = np.min(areas)
        if m < 1e-12:
            return 0.0
        # Shift for numerical stability
        shifted = areas - m
        exp_terms = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return m
        return m - (1.0 / beta) * np.log(sum_exp)

    def objective(params, beta):
        """
        Objective function to minimize: - (soft_min_area / hull_area)
        params: array of shape (26,)
        """
        # Reshape to (13, 2)
        points = params.reshape((n, 2))
        
        # Check for validity
        if not np.all(np.isfinite(points)):
            return 1e10
            
        hull_area = get_hull_area(points)
        if hull_area < 1e-9:
            return 1e10
            
        areas = get_areas(points)
        min_area_soft = soft_min(areas, beta)
        
        if min_area_soft < 1e-12:
            return 1e10
            
        ratio = min_area_soft / hull_area
        return -ratio

    def run_optimization(initial_points, beta_schedule, max_iter_per_beta=2000):
        """
        Run L-BFGS-B optimization with a schedule of increasing beta.
        """
        current_params = initial_points.flatten()
        best_params = current_params.copy()
        best_score = -1e10
        
        for beta in beta_schedule:
            # Check time
            if time.time() - start_time > time_limit - 30:
                break
                
            try:
                res = scipy.optimize.minimize(
                    objective, 
                    current_params, 
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': max_iter_per_beta, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                # Evaluate the hard metric (or very high beta soft-min) for this candidate
                cand_points = res.x.reshape((n, 2))
                hull_area = get_hull_area(cand_points)
                if hull_area < 1e-9:
                    continue
                
                areas = get_areas(cand_points)
                min_area = np.min(areas)
                score = min_area / hull_area
                
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
                    
                # Use the result as the starting point for the next beta
                current_params = res.x.copy()
                
            except Exception as e:
                # If optimization fails, keep current best
                pass

        return best_params, best_score

    # Initial configurations
    rng = np.random.default_rng(seed=42)
    
    # 1. Regular 13-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init1 = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # 2. Regular 13-gon scaled and shifted (should be same due to scale invariance, but helps with floating point)
    init2 = init1 * 2.0 + 1.0
    
    # 3. Random points in a square
    init3 = rng.uniform(-1, 1, (n, 2))
    
    # 4. Random points in a disk
    r = np.sqrt(rng.uniform(0, 1, n))
    theta = rng.uniform(0, 2 * np.pi, n)
    init4 = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    
    # 5. "Jittered" regular polygon
    init5 = init1 + rng.normal(0, 0.1, (n, 2))
    
    # 6. Square grid with center? (13 points: 3x3 grid + 4 corners? No, 13 is prime-ish)
    # Let's try a 4x4 grid minus 3 points
    grid = np.array([[x, y] for x in [-1.5, -0.5, 0.5, 1.5] for y in [-1.5, -0.5, 0.5, 1.5]])
    # Remove 3 points to get 13
    init6 = grid[:13]
    
    initializations = [init1, init2, init3, init4, init5, init6]
    
    best_global_params = None
    best_global_score = -1e10
    
    # Beta schedule: start low (smooth), end high (sharp)
    beta_schedule = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    
    for init in initializations:
        if time.time() - start_time > time_limit - 60:
            break
            
        params, score = run_optimization(init, beta_schedule)
        
        if score > best_global_score:
            best_global_score = score
            best_global_params = params.copy()
            
    # If we still have time, do a final polish with Nelder-Mead on the hard objective
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30 and best_global_params is not None:
        def neg_hard_objective(params):
            points = params.reshape((n, 2))
            if not np.all(np.isfinite(points)):
                return 1e10
            hull_area = get_hull_area(points)
            if hull_area < 1e-9:
                return 1e10
            areas = get_areas(points)
            min_area = np.min(areas)
            if min_area < 1e-12:
                return 1e10
            return -(min_area / hull_area)
        
        try:
            res_final = scipy.optimize.minimize(
                neg_hard_objective, 
                best_global_params, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            final_points = res_final.x.reshape((n, 2))
            hull_area = get_hull_area(final_points)
            if hull_area > 1e-9:
                areas = get_areas(final_points)
                min_area = np.min(areas)
                final_score = min_area / hull_area
                
                if final_score > best_global_score:
                    best_global_score = final_score
                    best_global_params = res_final.x.copy()
                    
        except Exception:
            pass

    # Final check and return
    if best_global_params is None:
        # Fallback to regular polygon
        best_global_params = init1.flatten()
        
    final_points = best_global_params.reshape((n, 2))
    
    # Ensure validity
    if not np.all(np.isfinite(final_points)):
        final_points = init1
        
    return final_points
