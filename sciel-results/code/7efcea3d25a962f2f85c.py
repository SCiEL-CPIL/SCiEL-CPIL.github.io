import numpy as np
from scipy.optimize import minimize

def construct():
    N = 16
    
    # Objective function to minimize: (max_dist / min_dist)^2
    def objective(params):
        points = params.reshape(N, 2)
        
        # Center the points
        points = points - np.mean(points, axis=0)
        
        # Normalize scale to avoid flat directions in the loss landscape
        # Fix sum of squared norms to 1
        norm_sq = np.sum(points**2)
        if norm_sq < 1e-12:
            return 1e10
        points = points / np.sqrt(norm_sq)
        
        # Compute pairwise distances
        # Use broadcasting for efficiency
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Exclude self-distances (diagonal)
        np.fill_diagonal(dists_sq, 1e10) # Set diagonal to large value so min ignores it
        min_dist_sq = np.min(dists_sq)
        max_dist_sq = np.max(dists_sq)
        
        if min_dist_sq < 1e-12:
            return 1e10
            
        ratio_sq = (np.sqrt(max_dist_sq) / np.sqrt(min_dist_sq))**2
        return ratio_sq

    # Initial guess: Points arranged in a circle
    # This is a reasonable starting point that has distinct points and finite ratio
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    init_params = init_points.flatten()
    
    # Optimize
    # Nelder-Mead is robust for small non-linear problems
    result = minimize(objective, init_params, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    # Extract optimized points
    opt_params = result.x
    opt_points = opt_params.reshape(N, 2)
    
    # Final cleanup: center and return
    opt_points = opt_points - np.mean(opt_points, axis=0)
    
    return opt_points
