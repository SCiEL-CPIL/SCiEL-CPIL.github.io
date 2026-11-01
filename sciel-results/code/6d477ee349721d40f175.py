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

def _soft_log_ratio(dist, tau):
    """
    Smooth approximation of log(max_d) - log(min_d) using soft min/max.
    This is a smooth surrogate for the actual objective.
    """
    mask = ~np.eye(N, dtype=bool)
    d = dist[mask]
    
    # Stable soft max: tau * log(sum(exp(d/tau)))
    max_d = np.max(d)
    soft_max = max_d + tau * np.log(np.sum(np.exp((d - max_d) / tau)))
    
    # Stable soft min: -tau * log(sum(exp(-d/tau)))
    min_d = np.min(d)
    soft_min = min_d - tau * np.log(np.sum(np.exp((min_d - d) / tau)))
    
    return soft_max - soft_min

def _objective(x, tau=0.1):
    """
    Objective function to minimize.
    Uses a soft log-ratio to encourage uniformity of pairwise distances.
    """
    pts = x.reshape(N, D)
    
    # Center the points to remove translation invariance
    pts = pts - np.mean(pts, axis=0)
    
    d = _pairwise_distances(pts)
    
    # Avoid log(0) and penalize close points
    if np.any(d < 1e-12):
        return 1e10
        
    return _soft_log_ratio(d, tau)

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
    return np.random.randn(N, D)

def construct():
    t_start = time.time()
    
    best_score = np.inf
    best_pts = None
    
    # Initial candidates
    candidates = []
    candidates.append(_generate_fib_sphere_points())
    candidates.append(np.random.randn(N, D))
    
    max_restarts = 200
    current_restarts = 2
    
    while current_restarts < max_restarts:
        if time.time() - t_start > DEADLINE:
            break
            
        # Alternate between Gaussian and Uniform-ish starts
        if current_restarts % 2 == 0:
            init_pts = np.random.randn(N, D)
        else:
            init_pts = np.random.randn(N, D)
            norms = np.linalg.norm(init_pts, axis=1)
            init_pts = init_pts / norms[:, np.newaxis]
            init_pts *= np.random.uniform(0.5, 1.5, N)[:, np.newaxis]
            
        current_restarts += 1
        
        try:
            init_pts = init_pts - np.mean(init_pts, axis=0)
            
            res = minimize(
                _objective, 
                init_pts.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            opt_pts = res.x.reshape(N, D)
            opt_pts = opt_pts - np.mean(opt_pts, axis=0)
            
            d = _pairwise_distances(opt_pts)
            if d.min() < 1e-6:
                continue
                
            score = _true_score(opt_pts)
            
            if score < best_score:
                best_score = score
                best_pts = opt_pts
                
        except Exception:
            continue

    if best_pts is None:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
        
    return best_pts
