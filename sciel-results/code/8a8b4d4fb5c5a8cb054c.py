# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    
    # Initial guess: 4x4 square grid
    # This is a known strong baseline for N=16
    x = np.linspace(0, 3, 4)
    y = np.linspace(0, 3, 4)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.ravel(), Y.ravel()))
    
    # Center the points to remove translational invariance
    points -= points.mean(axis=0)
    
    # Objective function: Minimize negative of (d_min / d_max)^2
    def objective(flat_points):
        pts = flat_points.reshape(n, 2)
        # Re-center to prevent drift during optimization
        pts -= pts.mean(axis=0)
        
        # Calculate pairwise distances
        dists = pdist(pts)
        
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return 0.0
            
        ratio = (d_min / d_max) ** 2
        return -ratio

    initial_flat = points.ravel()
    
    # Use Nelder-Mead for derivative-free optimization of non-smooth function
    # 500 iterations is usually sufficient for convergence in this low-dim space
    # and well within the time limit
    res = minimize(
        objective, 
        initial_flat, 
        method='Nelder-Mead', 
        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    
    best_points = res.x.reshape(n, 2)
    
    # Ensure the output is a valid finite array
    if not np.all(np.isfinite(best_points)):
         # Fallback to grid if optimization fails
         return points
         
    return best_points


# EVOLVE-BLOCK-END
