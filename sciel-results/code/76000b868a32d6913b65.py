import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _generate_test_points(n):
    # Use Fibonacci lattice for test points to ensure good coverage
    return _fibonacci_points(n)

def _covering_radius_grid(pts, n_grid=200):
    """Compute covering radius via dense grid (for final evaluation)."""
    pts = _normalize(pts)
    thetas = np.linspace(0, np.pi, n_grid)
    phis = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    THETA, PHI = np.meshgrid(thetas, phis)
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid = _normalize(grid)
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1, 1)
    max_dots = np.max(dots, axis=1)
    min_angles = np.arccos(max_dots)
    return np.max(min_angles)

def _soft_min_max_obj(x, test_pts, beta=20.0, gamma=20.0):
    """
    Objective: SoftMax over test points of SoftMin over candidate points of angular distance.
    x: flattened (N*3)
    """
    pts = x.reshape(N, 3)
    # Normalize points for distance calculation, but keep x as is for gradient?
    # If we normalize inside, the gradient w.r.t x is not just the gradient w.r.t normalized pts.
    # For numerical gradient, this is fine.
    pts_norm = _normalize(pts)
    
    # Compute distances
    # dots: (M, N)
    dots = test_pts @ pts_norm.T
    dots = np.clip(dots, -1, 1)
    dists = np.arccos(dots) # (M, N)
    
    # SoftMin over j (axis=1)
    # min(dists) approx -1/gamma * log(sum(exp(-gamma * dists)))
    m = np.min(dists, axis=1)
    soft_min = m - (1.0/gamma) * np.log(np.sum(np.exp(-gamma * (dists - m)), axis=1))
    
    # SoftMax over i (axis=0)
    # max(soft_min) approx 1/beta * log(sum(exp(beta * soft_min)))
    m2 = np.max(soft_min)
    soft_max = m2 + (1.0/beta) * np.log(np.sum(np.exp(beta * (soft_min - m2))))
    
    # Penalty for norm deviation
    norms = np.linalg.norm(pts, axis=1)
    penalty = 10.0 * np.sum((norms - 1.0)**2)
    
    return soft_max + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Test points
    n_test = 1000
    test_pts = _generate_test_points(n_test)
    
    best_pts = _fibonacci_points(N)
    best_cr = _covering_radius_grid(best_pts)
    
    # Optimization
    x0 = best_pts.flatten()
    
    # Try a few restarts with different initializations if time permits
    # But L-BFGS-B can be slow. Let's try one main run first.
    
    try:
        res = minimize(
            _soft_min_max_obj, 
            x0, 
            args=(test_pts,),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        
        if time.time() < deadline:
            pts_opt = res.x.reshape(N, 3)
            pts_opt = _normalize(pts_opt)
            cr_opt = _covering_radius_grid(pts_opt)
            
            if cr_opt < best_cr:
                best_cr = cr_opt
                best_pts = pts_opt
    except Exception:
        pass
    
    # If time remains, do a local refinement with a simpler method or more iterations?
    # Let's just return the best found.
    
    return _normalize(best_pts)

# EVOLVE-BLOCK-END
