import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import itertools

N = 14
D = 3
DEADLINE = 110.0  # Safety margin from 120s

def _pairwise_distances(pts):
    """Compute pairwise distances between all points."""
    # cdist is efficient
    d = cdist(pts, pts)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    return d[mask]

def _true_score(pts):
    """Calculate the actual score: (max/min)^2."""
    d = _pairwise_distances(pts)
    if len(d) == 0:
        return np.inf
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _objective(x):
    """
    Objective function to minimize.
    We minimize the variance of the log-distances.
    This encourages all distances to be equal, which minimizes the max/min ratio.
    """
    pts = x.reshape(N, D)
    
    # Center the points to remove translation invariance
    pts = pts - np.mean(pts, axis=0)
    
    # Optional: Normalize scale to prevent numerical issues, 
    # though variance of log-distances is scale invariant.
    # d_ij scale by s -> log(s*d) = log s + log d. Variance is unchanged.
    
    d = _pairwise_distances(pts)
    
    # Avoid log(0)
    if np.any(d < 1e-12):
        return 1e10
        
    log_d = np.log(d)
    
    # Variance of log distances
    var = np.var(log_d)
    
    return var

def _generate_random_points():
    """Generate random points in a unit sphere."""
    pts = np.random.randn(N, D)
    # Normalize to unit sphere surface? No, volume is better for packing.
    # Let's just use random normal distribution.
    return pts

def _generate_fib_sphere_points():
    """Generate points on a sphere using Fibonacci distribution."""
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))

def construct():
    t_start = time.time()
    
    best_score = np.inf
    best_pts = None
    
    # Initial candidates: Random and Fibonacci sphere
    candidates = [
        _generate_fib_sphere_points(),
        np.random.randn(N, D)
    ]
    
    # Add more random restarts if time permits
    max_restarts = 50
    current_restarts = 2
    
    while current_restarts < max_restarts:
        if time.time() - t_start > DEADLINE:
            break
        candidates.append(np.random.randn(N, D))
        current_restarts += 1
    
    # Optimize each candidate
    for i, init_pts in enumerate(candidates):
        if time.time() - t_start > DEADLINE:
            break
            
        # Center initial points
        init_pts = init_pts - np.mean(init_pts, axis=0)
        
        try:
            # Use L-BFGS-B for local optimization
            res = minimize(
                _objective, 
                init_pts.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            opt_pts = res.x.reshape(N, D)
            # Center the optimized points
            opt_pts = opt_pts - np.mean(opt_pts, axis=0)
            
            # Check for distinct points
            d = _pairwise_distances(opt_pts)
            if d.min() < 1e-6:
                continue
                
            score = _true_score(opt_pts)
            
            if score < best_score:
                best_score = score
                best_pts = opt_pts
                
        except Exception:
            continue

    # If no valid points were found (unlikely), fallback to grid
    if best_pts is None:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
        
    return best_pts
