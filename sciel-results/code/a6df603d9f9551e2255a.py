import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import itertools

N = 14
D = 3
DEADLINE = 115.0  # Safety margin from 120s

def _pairwise_distances(pts):
    """Compute pairwise distances between all points."""
    d = cdist(pts, pts)
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
    Minimizes the variance of the log-distances.
    This encourages all pairwise distances to be as similar as possible.
    """
    pts = x.reshape(N, D)
    
    # Center the points to remove translation invariance
    pts = pts - np.mean(pts, axis=0)
    
    d = _pairwise_distances(pts)
    
    # Avoid log(0) and penalize close points
    if np.any(d < 1e-12):
        return 1e10
        
    log_d = np.log(d)
    
    # Variance of log distances
    var = np.var(log_d)
    
    return var

def _generate_fib_sphere_points():
    """Generate points on a sphere using Fibonacci distribution."""
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))

def _generate_random_points():
    """Generate random points in a unit sphere."""
    # Random points in a sphere
    u = np.random.uniform(0, 1, (N, D))
    # Map to sphere volume
    r = np.random.uniform(0, 1, N) ** (1/3)
    # Actually, just Gaussian is often better for L-BFGS start
    return np.random.randn(N, D)

def construct():
    t_start = time.time()
    
    best_score = np.inf
    best_pts = None
    
    # Initial candidates
    # 1. Fibonacci sphere
    # 2. Random Gaussian
    # 3. Random Uniform in sphere
    
    candidates = []
    candidates.append(_generate_fib_sphere_points())
    candidates.append(np.random.randn(N, D))
    
    # Add more random restarts
    # We will generate and optimize on the fly to save memory and time
    # L-BFGS-B is fast for N=14, D=3 (42 variables)
    
    max_restarts = 200
    current_restarts = 2
    
    # We can run more restarts if time permits
    # Estimate time per restart: L-BFGS-B with 200 iterations on 42 vars is very fast (< 1ms usually)
    # So 200 restarts should take < 1s. We can do many more.
    
    while current_restarts < max_restarts:
        if time.time() - t_start > DEADLINE:
            break
            
        # Alternate between Gaussian and Uniform-ish starts
        if current_restarts % 2 == 0:
            init_pts = np.random.randn(N, D)
        else:
            # Random points on a sphere with random radius
            init_pts = np.random.randn(N, D)
            norms = np.linalg.norm(init_pts, axis=1)
            # Normalize to sphere surface then scale
            init_pts = init_pts / norms[:, np.newaxis]
            init_pts *= np.random.uniform(0.5, 1.5, N)[:, np.newaxis]
            
        current_restarts += 1
        
        try:
            # Center initial points
            init_pts = init_pts - np.mean(init_pts, axis=0)
            
            # Use L-BFGS-B for local optimization
            res = minimize(
                _objective, 
                init_pts.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
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

    # If we have time, do a second pass with more iterations on the best found so far?
    # Or just keep generating random restarts.
    
    # If no valid points were found (unlikely), fallback to grid
    if best_pts is None:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
        
    return best_pts
