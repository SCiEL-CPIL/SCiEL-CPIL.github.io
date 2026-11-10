# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates.

    """
    n = 16

    def objective(flat_points):
        pts = flat_points.reshape(n, 2)
        
        # Center
        pts = pts - np.mean(pts, axis=0)
        
        # Calculate initial max distance to normalize scale
        dists = pdist(pts)
        d_max = np.max(dists)
        
        if d_max < 1e-10:
            return 0.0
            
        # Scale so that d_max = 1
        pts = pts / d_max
        
        # Recalculate distances
        dists = pdist(pts)
        d_min = np.min(dists)
        
        # We want to maximize d_min, so minimize -d_min
        return -d_min

    # Define bounds for the initial search space.
    # Since we normalize, the absolute scale doesn't matter for the objective value,
    # but it helps DE explore. [-1, 1] is a reasonable starting box.
    bounds = [(-1.0, 1.0) for _ in range(n * 2)]
    
    # Run Differential Evolution
    # popsize=20 is a good balance for 32D
    result = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=1000, 
        tol=1e-12, 
        popsize=20,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=True # Uses Nelder-Mead to polish the best result
    )
    
    best_points = result.x.reshape(n, 2)
    
    # Final normalization for consistency (optional, but good for output stability)
    best_points -= np.mean(best_points, axis=0)
    dists = pdist(best_points)
    d_max = np.max(dists)
    if d_max > 1e-10:
        best_points = best_points / d_max
        
    return best_points

# EVOLVE-BLOCK-END
