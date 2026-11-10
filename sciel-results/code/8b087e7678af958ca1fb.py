# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses Differential Evolution to globally optimize the configuration.
    """
    n = 14
    d = 3
    
    def objective(x_flat):
        points = x_flat.reshape(n, d)
        # Normalize points to prevent scale drift and ensure finite values
        # Centering and scaling
        points = points - np.mean(points, axis=0, keepdims=True)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero if a point is at origin (unlikely but possible)
        norms[norms == 0] = 1e-10
        points = points / norms # Project onto unit sphere? 
        # Actually, forcing onto unit sphere might be too restrictive if the optimal shape isn't spherical?
        # But for min/max ratio, spherical codes are optimal. 
        # Let's just scale the whole set to have max distance 1.
        dists = pdist(points)
        if len(dists) == 0:
            return 1e10
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 1e10
        ratio = d_min / d_max
        if ratio <= 0:
            return 1e10
        return -np.log(ratio)

    # Initial bounds for differential evolution
    # We can start with a wide range
    bounds = [(-1.0, 1.0)] * (n * d)
    
    # Run differential evolution
    # maxiter and popsize control the thoroughness. 
    # For 42 dimensions, we need a decent population.
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            seed=42,
            tol=1e-8,
            polish=True # Use Nelder-Mead to polish the result
        )
        best_points = result.x.reshape(n, d)
    except Exception:
        # Fallback to random if optimization fails
        np.random.seed(42)
        best_points = np.random.randn(n, d)
        
    # Final normalization to ensure stability and consistent scale
    best_points = best_points - np.mean(best_points, axis=0, keepdims=True)
    dists = pdist(best_points)
    d_max = np.max(dists)
    if d_max > 0:
        best_points = best_points / d_max
        
    return best_points

# EVOLVE-BLOCK-END
