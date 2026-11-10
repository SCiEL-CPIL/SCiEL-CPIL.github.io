# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    
    # Fix point 0 at (0,0) and point 1 at (1,0) to remove translational, rotational, and scaling symmetries.
    # This reduces the search space from 32 variables to 28.
    n_free = n - 2
    dim = 2 * n_free
    
    def objective(flat_free_points):
        free_points = flat_free_points.reshape(n_free, 2)
        full_points = np.vstack([(0.0, 0.0), (1.0, 0.0), free_points])
        
        dists = pdist(full_points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        ratio = d_min / d_max
        return -ratio * ratio

    # Bounds for the free points.
    # Since the distance between fixed points is 1, and we want to maximize the min distance,
    # points shouldn't be too far away. A bound of [-4, 4] is safe for 16 points.
    bounds = [(-4.0, 4.0)] * dim
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=2000, 
            popsize=30, 
            tol=1e-10, 
            seed=42,
            polish=True
        )
        best_flat = res.x
    except Exception:
        # Fallback: simple random initialization if DE fails
        best_flat = np.random.uniform(-1, 1, dim)
        
    best_free_points = best_flat.reshape(n_free, 2)
    best_points = np.vstack([(0.0, 0.0), (1.0, 0.0), best_free_points])
    
    return best_points
# EVOLVE-BLOCK-END
