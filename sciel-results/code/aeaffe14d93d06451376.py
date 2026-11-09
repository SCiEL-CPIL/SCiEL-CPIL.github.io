# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, basinhopping

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _objective(x):
    """
    Objective function for minimization.
    We want to maximize d_min / d_max.
    This is equivalent to minimizing -log(d_min / d_max) = log(d_max) - log(d_min).
    Since d_max is usually constrained by the spread, and we are on a sphere,
    we can just focus on maximizing d_min by minimizing -log(d_min).
    However, to be robust against scale, we compute the ratio.
    Using log makes the gradient smoother and more stable for optimization.
    """
    points = x.reshape(-1, 3)
    if not np.all(np.isfinite(points)):
        return 1e6
    
    dists = pdist(points)
    if len(dists) == 0:
        return 1e6
        
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_min <= 1e-12 or d_max <= 1e-12:
        return 1e6
        
    # We want to maximize ratio, so minimize negative log ratio
    # ratio = d_min / d_max
    # -log(ratio) = log(d_max) - log(d_min)
    return np.log(d_max) - np.log(d_min)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 620  # Safety margin
    
    best_score = 0.0
    best_points = None
    n = 14
    
    # Initial guess: Fibonacci sphere
    def fibonacci_sphere(n):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - 2 * (i + 0.5) / n
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)
    
    # We use Basin Hopping to explore the configuration space.
    # The local optimizer (Powell) will refine each step.
    # We run multiple independent basin hopping attempts from different starts.
    
    num_starts = 5
    maxfun = 2000  # Max function evaluations per local minimization step
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 30:
            break
        
        # Different initializations
        if start_idx == 0:
            x0 = fibonacci_sphere(n).flatten()
        else:
            # Random points on sphere, flattened
            x0 = np.random.randn(n, 3)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            x0 = x0.flatten()
        
        try:
            # Basin hopping with Powell local minimizer
            # stepsize controls the perturbation magnitude
            res = basinhopping(
                _objective, 
                x0, 
                niter=50, 
                stepsize=0.5, 
                minimizer_kwargs={'method': 'Powell', 'options': {'maxiter': 2000, 'maxfev': 2000}},
                seed=start_idx
            )
            
            if res.success or res.fun < 100: # Check if it converged reasonably
                points = res.x.reshape(-1, 3)
                # Normalize to unit sphere to ensure consistent scale for scoring
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                
                s = _score(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
                    
        except Exception:
            continue
            
    # If we haven't found anything good, fallback to fibonacci
    if best_points is None:
        best_points = fibonacci_sphere(n)
    else:
        # Final polish: run a few more Powell steps on the best result
        try:
            x_best = best_points.flatten()
            res = minimize(_objective, x_best, method='Powell', options={'maxiter': 5000, 'maxfev': 5000})
            if res.fun < 100:
                points = res.x.reshape(-1, 3)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                s = _score(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
