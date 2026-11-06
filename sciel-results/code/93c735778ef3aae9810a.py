"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _fibonacci_spiral(n, offset=0.5):
    """Generate initial Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return pts / norms

def _covering_radius_soft(flat_pts, test_points, beta=10.0):
    """
    Compute a smooth approximation of the covering radius.
    flat_pts: (N*3,) array of flattened points.
    test_points: (M, 3) array of test points on the sphere.
    beta: sharpness parameter for the soft-max.
    """
    pts = flat_pts.reshape(N, 3)
    pts = _normalize(pts)
    
    # Compute dot products between test points and configuration points
    # test_points: (M, 3), pts: (N, 3)
    dots = test_points @ pts.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    
    # Angular distances
    dists = np.arccos(dots)  # (M, N)
    
    # Minimum distance from each test point to any configuration point
    min_dists = np.min(dists, axis=1)  # (M,)
    
    # Soft max approximation of the maximum of min_dists
    # max(x) ~ (1/beta) * log(sum(exp(beta*x)))
    # To avoid overflow, subtract the max
    max_d = np.max(min_dists)
    if beta <= 0:
        return max_d
    shifted = min_dists - max_d
    # Use log-sum-exp for stability
    log_sum = max_d + np.log(np.sum(np.exp(shifted)))
    return log_sum / beta

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    # Initial configuration
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    
    # Define test points for the objective function
    # We use a Fibonacci spiral for test points as well, as it's uniform
    n_test_coarse = 5000
    test_points_coarse = _fibonacci_spiral(n_test_coarse)
    
    # Initial objective
    # We use a soft-max to make the objective differentiable
    # Start with a lower beta for smoother landscape
    beta_coarse = 5.0
    
    def objective_coarse(flat_pts):
        return _covering_radius_soft(flat_pts, test_points_coarse, beta_coarse)
    
    # Phase 1: Coarse optimization
    try:
        res = minimize(
            objective_coarse,
            pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-5, 'maxfun': 200}
        )
        if res.fun < objective_coarse(best_pts.flatten()):
            best_pts = _normalize(res.x.reshape(N, 3))
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        return best_pts
    
    # Phase 2: Fine optimization with more test points and higher beta
    n_test_fine = 20000
    test_points_fine = _fibonacci_spiral(n_test_fine)
    beta_fine = 20.0
    
    def objective_fine(flat_pts):
        return _covering_radius_soft(flat_pts, test_points_fine, beta_fine)
    
    try:
        res2 = minimize(
            objective_fine,
            best_pts.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 300, 'gtol': 1e-6, 'maxfun': 300}
        )
        # Compare using the same fine objective
        if res2.fun < objective_fine(best_pts.flatten()):
            best_pts = _normalize(res2.x.reshape(N, 3))
    except Exception:
        pass
    
    # Phase 3: Very fine polish if time permits
    if time.time() < deadline - 10:
        n_test_very_fine = 50000
        test_points_very_fine = _fibonacci_spiral(n_test_very_fine)
        beta_very_fine = 50.0
        
        def objective_very_fine(flat_pts):
            return _covering_radius_soft(flat_pts, test_points_very_fine, beta_very_fine)
        
        try:
            res3 = minimize(
                objective_very_fine,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-7, 'maxfun': 100}
            )
            if res3.fun < objective_very_fine(best_pts.flatten()):
                best_pts = _normalize(res3.x.reshape(N, 3))
        except Exception:
            pass
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
