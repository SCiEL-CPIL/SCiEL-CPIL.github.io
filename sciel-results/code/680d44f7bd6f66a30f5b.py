import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 16
D = 2

def compute_score(points):
    """
    Compute the squared ratio of max to min pairwise distance.
    points: (N, D) array
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    off_diag_dist_sq = dist_sq[mask]
    
    if len(off_diag_dist_sq) == 0:
        return np.inf
        
    min_dist_sq = np.min(off_diag_dist_sq)
    max_dist_sq = np.max(off_diag_dist_sq)
    
    if min_dist_sq < 1e-12:
        return np.inf
        
    ratio_sq = max_dist_sq / min_dist_sq
    return ratio_sq

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin for 120s limit
    
    # Initial guess: Grid
    x = np.arange(4)
    y = np.arange(4)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    grid_points -= np.mean(grid_points, axis=0)
    
    def objective(params):
        points = params.reshape(N, D)
        return compute_score(points)
    
    bounds = [(-5.0, 5.0) for _ in range(N * D)]
    
    best_result = None
    best_score = np.inf
    
    try:
        # Differential Evolution
        # Using latinhypercube for good spread
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=30, 
            seed=42, 
            tol=1e-8,
            init='latinhypercube'
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_result = result.x
            
        # Refine with Nelder-Mead
        if best_result is not None and (time.time() - start_time) < timeout - 5.0:
            res_nm = minimize(
                objective, 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_result = res_nm.x
                
    except Exception:
        pass

    if best_result is None:
        return grid_points

    points = best_result.reshape(N, D)
    points -= np.mean(points, axis=0)
    return points
