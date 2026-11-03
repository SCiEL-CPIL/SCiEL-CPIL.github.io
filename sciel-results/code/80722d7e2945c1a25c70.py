"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _generate_grid(n_theta=50, n_phi=50):
    """Generate a grid of points on the sphere for objective evaluation."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi)
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

# Pre-generate grid for efficiency
GRID_POINTS = _generate_grid(60, 60)

def _covering_radius_proxy(pts):
    """
    Compute a smooth proxy for the covering radius.
    Uses log-sum-exp of squared distances to approximate the max distance.
    """
    # pts: (N, 3)
    # grid: (M, 3)
    # Compute squared distances: ||g - p||^2 = 2 - 2*g.p
    # We want min over p for each g.
    # dist_sq[g, p] = 2 - 2 * (grid @ pts.T)
    
    dot_products = GRID_POINTS @ pts.T  # (M, N)
    dist_sq = 2.0 - 2.0 * dot_products  # (M, N)
    
    # Clip to ensure non-negative (numerical stability)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    # For each grid point, find min squared distance
    min_dist_sq = np.min(dist_sq, axis=1)  # (M,)
    
    # Approximate max(min_dist) using log-sum-exp
    # We want to maximize the coverage, so we minimize the max distance.
    # Let d = min_dist_sq. We want to minimize max(d).
    # Use a soft max: log(sum(exp(d / tau)) * tau)
    # Since d is squared distance, this is a proxy for angular distance squared.
    
    tau = 0.05
    # Scale to avoid overflow
    max_d = np.max(min_dist_sq)
    if max_d == 0:
        return 0.0
    
    # log-sum-exp trick
    shifted = min_dist_sq / tau
    shifted -= np.max(shifted)
    lse = np.log(np.sum(np.exp(shifted))) + np.max(shifted)
    
    # The proxy value is proportional to the effective max distance.
    # We just need a monotonic function.
    return lse * tau

def _local_optimize(pts, max_time=20.0):
    """
    Local optimization using scipy.optimize.minimize on the proxy function.
    Parameters are flattened points. We project back to sphere after each step.
    """
    start_time = time.time()
    x0 = pts.flatten()
    
    def objective(x):
        pts_candidate = x.reshape(N, 3)
        pts_candidate = _normalize(pts_candidate)
        return _covering_radius_proxy(pts_candidate)
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        # Check time
        if time.time() - start_time > max_time:
            # Return current best if timed out? Nelder-Mead doesn't easily interrupt mid-iteration.
            # We rely on maxiter and the overall deadline check in construct().
            pass
            
        best_pts = _normalize(res.x.reshape(N, 3))
        return best_pts, objective(res.x)
        
    except Exception:
        return pts, _covering_radius_proxy(pts)

def construct():
    """Construct 50 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Start with Fibonacci spiral
    best_pts = _fibonacci_spiral()
    best_score = _covering_radius_proxy(best_pts)
    
    # Try a few random restarts
    np.random.seed(42)
    
    # We can do 2-3 restarts with Nelder-Mead as it can be slow.
    # Let's do 1 initial optimization on Fibonacci, then maybe 1 random restart.
    
    # 1. Optimize Fibonacci start
    if time.time() < deadline - 30.0:
        opt_pts, score = _local_optimize(best_pts, max_time=30.0)
        if score < best_score:
            best_score = score
            best_pts = opt_pts
    
    # 2. Random restart
    if time.time() < deadline - 30.0:
        pts_rand = np.random.randn(N, 3)
        pts_rand = _normalize(pts_rand)
        opt_pts, score = _local_optimize(pts_rand, max_time=30.0)
        if score < best_score:
            best_score = score
            best_pts = opt_pts

    # 3. Final polish with smaller tolerance if time permits
    if time.time() < deadline - 10.0:
        opt_pts, score = _local_optimize(best_pts, max_time=20.0)
        if score < best_score:
            best_score = score
            best_pts = opt_pts

    return best_pts
# EVOLVE-BLOCK-END
