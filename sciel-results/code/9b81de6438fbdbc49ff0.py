"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    n = points.shape[0]
    dists = pdist(points)
    if len(dists) == 0:
        return 1e10
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-10:
        return 1e10
    return (max_d / min_d) ** 2

def construct():
    n = N
    d = D
    
    def objective(x):
        """
        Objective function for differential evolution.
        x: flattened array of shape (n*d,)
        Returns the squared ratio of max to min distance.
        """
        points = x.reshape(n, d)
        # Calculate pairwise distances
        dists = pdist(points)
        
        # Check for collisions (min distance too small)
        min_d = np.min(dists)
        if min_d < 1e-8:
            return 1e10 + 1.0 / (min_d + 1e-10)
            
        max_d = np.max(dists)
        
        # The ratio is scale-invariant, so we don't need to constrain the scale.
        # However, DE might drift to very large or small values.
        # We can normalize inside the objective to keep numbers stable, 
        # but the ratio itself is invariant.
        # Let's just return the squared ratio.
        ratio = max_d / min_d
        return ratio ** 2

    # Define bounds for each coordinate
    # The optimal configuration is typically within a unit sphere or cube.
    # Using [-1, 1] is a reasonable bound. If the points go outside, 
    # the ratio won't improve significantly because relative positions matter.
    bounds = [(-1.0, 1.0)] * (n * d)
    
    # Run Differential Evolution
    # maxiter: 1000 might be too slow if popsize is large. 
    # popsize: 15 * dim is standard. dim = 42. popsize = 630.
    # This is a large population. Let's try popsize=20 and maxiter=500.
    # Time limit is 120s.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=300, 
            popsize=15, 
            tol=1e-8, 
            atol=1e-8,
            polish=True, # Use L-BFGS-B to polish the result
            init='sobol'
        )
        
        if result.success or result.fun < 1e9:
            best_points = result.x.reshape(n, d)
            return best_points
        else:
            # Fallback to best found even if not "successful"
            return result.x.reshape(n, d)
            
    except Exception as e:
        # Fallback to random if DE fails
        return np.random.randn(n, d)

# EVOLVE-BLOCK-END
