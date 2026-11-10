# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a Fibonacci lattice initialization and local optimization (Nelder-Mead) 
    to maximize the minimum pairwise distance.
    """
    n = 14
    d = 3
    
    # 1. Initialization: Fibonacci Lattice (Spiral) on the unit sphere
    # This provides a well-distributed set of points, much better than random.
    i = np.arange(n)
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    # Golden angle
    angle = 2 * np.pi / (phi ** 2)
    # Z coordinates evenly spaced
    z = 1 - 2 * (i + 0.5) / n
    r = np.sqrt(1 - z**2)
    theta = angle * i
    
    points = np.zeros((n, d))
    points[:, 0] = r * np.cos(theta)
    points[:, 1] = r * np.sin(theta)
    points[:, 2] = z
    
    # 2. Objective function: Maximize the minimum pairwise distance
    # We minimize the negative of the minimum distance.
    def objective(x):
        p = x.reshape(n, d)
        # Compute pairwise distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        # Get upper triangle indices to avoid self-distance and duplicates
        i_upper = np.triu_indices(n, k=1)
        min_dist_sq = dist_sq[i_upper].min()
        # Return negative min distance (to minimize)
        return -np.sqrt(min_dist_sq)
    
    # 3. Optimization
    # Nelder-Mead is a derivative-free method suitable for non-smooth functions like min distance.
    x0 = points.flatten()
    
    # Run optimization
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    optimal_points = res.x.reshape(n, d)
    
    # 4. Post-processing
    # Center the points
    optimal_points -= np.mean(optimal_points, axis=0, keepdims=True)
    
    # Scale so that max distance is 1 (optional, but keeps numbers manageable)
    diff = optimal_points[:, np.newaxis, :] - optimal_points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    max_dist = np.sqrt(np.max(dist_sq))
    
    if max_dist > 0:
        optimal_points = optimal_points / max_dist
        
    return optimal_points

# EVOLVE-BLOCK-END
