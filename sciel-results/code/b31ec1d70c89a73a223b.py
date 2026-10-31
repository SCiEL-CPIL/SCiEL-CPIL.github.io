import numpy as np
from scipy.optimize import differential_evolution

def construct():
    N = 16
    
    def objective(x):
        # x has length 2*(N-2)
        # Reconstruct points
        pts = np.zeros((N, 2))
        pts[0] = [0, 0]
        pts[1] = [1, 0]
        rest = x.reshape(N-2, 2)
        pts[2:] = rest
        
        # Compute pairwise distances
        dist_matrix = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        iu = np.triu_indices(N, 1)
        pairwise_dists = dist_matrix[iu]
        
        min_d = np.min(pairwise_dists)
        max_d = np.max(pairwise_dists)
        
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d) ** 2

    # Bounds for the remaining N-2 points
    # We need to choose bounds large enough to contain the optimal configuration
    # If min distance is 1 (fixed by P0-P1), max distance is likely < 10.
    # Let's use bounds [-5, 5] for the free points.
    bounds = [(-5.0, 5.0)] * (2 * (N - 2))
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            tol=1e-6, 
            seed=42,
            polish=True,
            init='sobol'
        )
        if res.fun < 1e9:
            # Reconstruct full points
            pts = np.zeros((N, 2))
            pts[0] = [0, 0]
            pts[1] = [1, 0]
            pts[2:] = res.x.reshape(N-2, 2)
            return pts
    except Exception:
        pass

    # Fallback
    return np.random.uniform(-5, 5, (N, 2))
