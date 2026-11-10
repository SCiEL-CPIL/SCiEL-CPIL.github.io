# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Initial guess: 4x4 grid
    # This configuration has d_min = 1, d_max = sqrt(18), R = 1/18 ~ 0.0555
    x = np.arange(4)
    y = np.arange(4)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Normalize initial points to have unit max distance? 
    # The objective is ratio, so scale doesn't matter, but it helps numerical stability.
    # Let's just use the raw grid.
    
    def objective(params):
        """
        Objective function to minimize: -(d_min / d_max)
        We maximize d_min/d_max, so we minimize its negative.
        """
        pts = params.reshape((n, 2))
        dists = pdist(pts, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-10:
            return 0.0 # Avoid division by zero, score 0
            
        ratio = d_min / d_max
        return -ratio

    best_points = points
    best_score = -objective(points)
    
    # Try local optimization from the grid start
    try:
        res = minimize(
            objective, 
            points.flatten(), 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() > deadline:
            return best_points
            
        candidate_points = res.x.reshape((n, 2))
        candidate_score = -objective(candidate_points)
        
        if candidate_score > best_score:
            best_points = candidate_points
            best_score = candidate_score
            
    except Exception:
        pass

    # If time allows, try a few random restarts or perturbations to escape local minima
    # The 4x4 grid is likely the global optimum or very close to it for N=16 in 2D.
    # However, a hexagonal packing fragment might be slightly better?
    # For 16 points, the optimal packing in a square is indeed a 4x4 grid (or close to it).
    # Let's try one more refinement with a different method or initial guess if time permits.
    
    if time.time() < deadline - 60:
        # Try a hexagonal-like arrangement
        # 4 rows of 4 points, but staggered
        # This is harder to set up correctly as a grid, so let's stick to the optimized grid.
        
        # Alternative: Random restarts
        np.random.seed(42)
        for i in range(5):
            if time.time() > deadline - 30:
                break
                
            # Random points in a square
            rand_pts = np.random.rand(n, 2) * 10 - 5
            try:
                res_rand = minimize(
                    objective, 
                    rand_pts.flatten(), 
                    method='Powell', 
                    options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-12}
                )
                if time.time() > deadline:
                    break
                    
                cand = res_rand.x.reshape((n, 2))
                score = -objective(cand)
                if score > best_score:
                    best_points = cand
                    best_score = score
            except Exception:
                continue

    return best_points

# EVOLVE-BLOCK-END
