import numpy as np
import time
import scipy.spatial
import scipy.optimize

N = 50
TIME_LIMIT = 135.0  # Safety margin from 150s

def _compute_circumradii_smooth(points, beta=50.0):
    """
    Compute soft-max of squared circumradii of the facets of the convex hull.
    Returns (soft_max_value, max_sq_rad).
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return 1e10, 1e10
    
    theta_sq_list = []
    
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        n = np.cross(v2 - v1, v3 - v1)
        norm_n = np.linalg.norm(n)
        
        if norm_n < 1e-12:
            theta_sq_list.append(1e10)
            continue
            
        c = n / norm_n
        dot = np.dot(c, v1)
        dot = np.clip(dot, -1.0, 1.0)
        
        if dot < 0:
            c = -c
            dot = -dot
            
        theta = np.arccos(dot)
        theta_sq = theta * theta
        theta_sq_list.append(theta_sq)
    
    if not theta_sq_list:
        return 1e10, 1e10
        
    theta_sq_arr = np.array(theta_sq_list)
    max_sq_rad = np.max(theta_sq_arr)
    
    # Soft-max via log-sum-exp
    shifted = beta * theta_sq_arr
    shifted -= np.max(shifted)
    soft_max = np.max(theta_sq_arr) + np.log(np.sum(np.exp(shifted))) / beta
    
    return soft_max, max_sq_rad

def _objective(x, beta=50.0):
    """
    Objective: smooth max of squared circumradii + sphere constraint penalty.
    """
    points = x.reshape(N, 3)
    
    norms = np.linalg.norm(points, axis=1)
    sphere_penalty = np.sum((norms - 1.0)**2) * 50.0
    
    soft_max, _ = _compute_circumradii_smooth(points, beta)
    
    return soft_max + sphere_penalty

def _local_optimize(start_points, beta=50.0, maxiter=200):
    """
    Run local optimization from a given start using L-BFGS-B.
    """
    x0 = start_points.flatten()
    
    try:
        res = scipy.optimize.minimize(
            _objective, 
            x0, 
            args=(beta,),
            method='L-BFGS-B', 
            options={'maxiter': maxiter, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-8}
        )
    except Exception:
        return start_points, _objective(x0, beta)
    
    x_opt = res.x.reshape(N, 3)
    x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
    
    _, max_sq_rad = _compute_circumradii_smooth(x_opt, beta=200.0)
    score = np.sqrt(max_sq_rad)
    
    return x_opt, score

def _polish(points, maxiter=300):
    """
    Final high-precision polish pass.
    """
    x0 = points.flatten()
    
    try:
        res = scipy.optimize.minimize(
            _objective, 
            x0, 
            args=(100.0,),
            method='L-BFGS-B', 
            options={'maxiter': maxiter, 'disp': False, 'ftol': 1e-15, 'gtol': 1e-10}
        )
    except Exception:
        return points
    
    x_opt = res.x.reshape(N, 3)
    x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
    
    return x_opt

def _fibonacci_sphere(n):
    """Generate points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng=None):
    """Generate random points on sphere."""
    if rng is None:
        rng = np.random.default_rng()
    pts = rng.standard_normal((n, 3))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def construct():
    start_time = time.time()
    best_points = None
    best_score = np.inf
    
    rng = np.random.default_rng(42)
    
    starts = []
    starts.append(_fibonacci_sphere(N))
    
    for _ in range(8):
        starts.append(_random_sphere(N, rng))
    
    fib = _fibonacci_sphere(N)
    for _ in range(3):
        perturbed = fib + 0.1 * rng.standard_normal((N, 3))
        perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
        starts.append(perturbed)
    
    for idx, start in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 15:
            break
        
        beta = 50.0 if idx < 5 else 80.0
        points, score = _local_optimize(start, beta=beta, maxiter=150)
        
        if score < best_score:
            best_score = score
            best_points = points
    
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 10 and best_points is not None:
        best_points = _polish(best_points, maxiter=200)
        _, max_sq_rad = _compute_circumradii_smooth(best_points, beta=200.0)
        best_score = np.sqrt(max_sq_rad)
    
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
