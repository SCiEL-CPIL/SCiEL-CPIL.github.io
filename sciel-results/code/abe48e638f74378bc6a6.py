import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution
import time

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    
    def objective(x_flat):
        """
        Objective function for Differential Evolution.
        Minimizes negative log-ratio.
        """
        points = x_flat.reshape(n, 2)
        
        # Normalize to unit norm to handle scale invariance
        # Centering helps prevent drift
        points = points - np.mean(points, axis=0)
        norm = np.linalg.norm(points)
        if norm < 1e-10:
            return 1e10 # Penalize degenerate solutions
            
        points = points / norm
        
        # Calculate distances
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_min < 1e-10 or d_max < 1e-10:
            return 1e10
            
        # We want to maximize (d_min / d_max)^2
        # Max log((d_min/d_max)^2) = 2 * (log(d_min) - log(d_max))
        # DE minimizes, so return negative
        return -2.0 * (np.log(d_min) - np.log(d_max))

    def run_de(seed):
        np.random.seed(seed)
        
        # Define bounds. Since we normalize, the scale doesn't matter,
        # but the relative positions do. A box of [-1, 1] or [-2, 2] is sufficient.
        # Larger bounds allow more initial diversity.
        bounds = [(-2.0, 2.0)] * (n * 2)
        
        # Run differential evolution
        # popsize: 15 is a good balance.
        # maxiter: 100 iterations.
        # polish=True uses L-BFGS-B at the end to refine the solution, which is crucial for precision.
        result = differential_evolution(
            objective,
            bounds,
            seed=seed,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            init='sobol',
            polish=True
        )
        
        return result.x, -result.fun

    best_points = None
    best_score = -np.inf
    
    # Total time budget
    total_time = 590
    start_total = time.time()
    
    # Run multiple DE instances with different seeds
    # DE is stochastic, so different seeds explore different regions.
    num_runs = 20
    
    for i in range(num_runs):
        if time.time() - start_total > total_time - 10:
            break
            
        seed = 42 + i
        try:
            x_opt, score_opt = run_de(seed)
            if score_opt > best_score:
                best_score = score_opt
                best_points = x_opt.reshape(n, 2)
        except Exception:
            continue
            
    # If best_points is still None (shouldn't happen), fallback to grid
    if best_points is None:
        x = np.arange(4) - 1.5
        y = np.arange(4) - 1.5
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack((xx.ravel(), yy.ravel()))
        
    return best_points
