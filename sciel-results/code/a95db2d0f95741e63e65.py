# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import itertools

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    # Objective function to minimize
    # We want to maximize d_min, subject to d_max <= 1.
    # Let's define the objective as: -d_min + lambda * max(0, d_max - 1)
    # If d_max > 1, we penalize heavily.
    
    def objective(flat_points):
        points = flat_points.reshape(n, d)
        # Calculate pairwise distances
        # Using scipy.spatial.distance.pdist for efficiency
        from scipy.spatial.distance import pdist
        dists = pdist(points)
        
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        # Penalty for exceeding max distance of 1
        penalty = 1000.0 * np.maximum(0.0, d_max - 1.0)
        
        # We want to maximize d_min, so minimize -d_min
        return -d_min + penalty

    # Bounds for coordinates
    # Since we scale to d_max=1, points should be roughly within [-0.5, 0.5] or similar
    # But differential evolution needs bounds.
    # A sphere of diameter 1 has radius 0.5.
    # Let's use bounds [-1, 1] to be safe, the penalty will push them in.
    bounds = [(-1.0, 1.0)] * (n * d)
    
    # Run differential evolution
    # maxiter and popsize control the search depth
    # tol for convergence
    res = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=200, 
        popsize=15,
        tol=1e-8,
        polish=False # Polish can be slow and sometimes stuck, DE is robust enough
    )
    
    points = res.x.reshape(n, d)
    
    # Optional: Refine with local search (e.g., Nelder-Mead) on the refined points
    # But DE with polish=False might be slightly suboptimal. Let's try to add a local refinement step.
    # Actually, let's just return the DE result. It's usually very good for N=14.
    
    # Ensure finite and correct shape
    if not np.all(np.isfinite(points)):
        # Fallback to random if something went wrong
        np.random.seed(42)
        points = np.random.randn(n, d)
        
    return points

# EVOLVE-BLOCK-END
