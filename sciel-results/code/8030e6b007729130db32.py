# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    def soft_min(dists, t=50.0):
        """Differentiable approximation of the minimum distance."""
        return -np.log(np.sum(np.exp(-dists / t)) / len(dists)) * t

    def objective(x):
        pts = x.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        # Normalize so d_max = 1
        dists_norm = dists / d_max
        d_min_norm = soft_min(dists_norm)
        return -d_min_norm  # Negative because we maximize

    # Initial guess: 4x4 grid, then optimize
    x = np.arange(4)
    y = np.arange(4)
    xx, yy = np.meshgrid(x, y)
    points_init = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Normalize initial points to have unit max distance
    dmax_init = np.max(pdist(points_init))
    points_init = points_init / dmax_init
    x0 = points_init.ravel()
    
    best_points = points_init
    best_score = objective(x0)
    
    # Try multiple random restarts with optimization
    np.random.seed(42)
    n_restarts = 10
    
    for i in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if i == 0:
            x_start = x0
        else:
            # Random initial configuration
            pts = np.random.randn(n, d)
            dmax_r = np.max(pdist(pts))
            if dmax_r > 1e-12:
                pts = pts / dmax_r
            x_start = pts.ravel()
        
        res = minimize(objective, x_start, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.success or res.fun < best_score:
            best_score = res.fun
            best_points = res.x.reshape(n, d)
        
        # If we're doing well, try more iterations with the best solution
        if best_score < -0.1 and time.time() < deadline - 60:
            res2 = minimize(objective, res.x, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res2.fun < best_score:
                best_score = res2.fun
                best_points = res2.x.reshape(n, d)
    
    return best_points


# EVOLVE-BLOCK-END
