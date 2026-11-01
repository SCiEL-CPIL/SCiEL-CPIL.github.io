import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 14
D = 3
DEADLINE = 130.0
_t0 = None

def _true_score(pts):
    """Compute (max/min pairwise distance)^2"""
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-12:
        return np.inf
    return (mx / mn) ** 2

def _objective(x):
    """Objective for optimization: minimize log(max/min) with penalties"""
    pts = x.reshape(N, D)
    pts = pts - pts.mean(axis=0)
    d = cdist(pts, pts)
    off_diag = d[~np.eye(N, dtype=bool)]
    mn = off_diag.min()
    mx = off_diag.max()
    if mn < 1e-6:
        return 1e8
    # Use log ratio to make it more tractable
    return np.log(mx) - np.log(mn)

def _fib_sphere(n, r=1.0):
    """Generate points on a sphere using Fibonacci lattice"""
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.column_stack([x, y, z])

def _random_sphere(n, r=1.0):
    """Generate random points on a sphere"""
    theta = np.random.uniform(0, 2 * np.pi, n)
    z = np.random.uniform(-1, 1, n)
    x = np.sqrt(1 - z**2) * np.cos(theta)
    y = np.sqrt(1 - z**2) * np.sin(theta)
    return r * np.column_stack([x, y, z])

def _optimize_from_start(start_pts, time_limit):
    """Optimize from a given starting configuration"""
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    x0 = start_pts.flatten()
    
    # Try multiple optimization methods
    best_result = None
    best_obj = _objective(x0)
    best_pts = start_pts.copy()
    
    methods = ['L-BFGS-B', 'Powell', 'Nelder-Mead']
    
    for method in methods:
        if time.time() - _t0 > time_limit:
            break
        
        try:
            result = minimize(_objective, x0, method=method, 
                             options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < best_obj:
                best_obj = result.fun
                best_pts = result.x.reshape(N, D)
                best_pts = best_pts - best_pts.mean(axis=0)
        except Exception:
            pass
    
    return best_pts, best_obj

def construct():
    global _t0
    _t0 = time.time()
    time_limit = DEADLINE - 10.0  # Safety margin
    
    best_pts = None
    best_score = np.inf
    
    # Strategy 1: Start from Fibonacci sphere
    fib_pts = _fib_sphere(N)
    fib_pts, fib_obj = _optimize_from_start(fib_pts, time_limit)
    fib_score = _true_score(fib_pts)
    if fib_score < best_score:
        best_score = fib_score
        best_pts = fib_pts.copy()
    
    # Strategy 2: Start from random sphere configurations
    n_random_starts = 5
    for i in range(n_random_starts):
        if time.time() - _t0 > time_limit:
            break
        
        np.random.seed(42 + i)
        rand_pts = _random_sphere(N)
        opt_pts, opt_obj = _optimize_from_start(rand_pts, time_limit)
        opt_score = _true_score(opt_pts)
        if opt_score < best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()
    
    # Strategy 3: Try some structured configurations
    # Icosahedron-like: 12 vertices of icosahedron + 2 more
    if time.time() - _t0 < time_limit:
        # Golden ratio
        phi = (1 + 5 ** 0.5) / 2
        ico = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
            [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
            [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
        ], dtype=float)
        ico = ico[:12]
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        
        # Add 2 points at north and south poles
        extra = np.array([[0, 0, 1], [0, 0, -1]], dtype=float)
        ico14 = np.vstack([ico, extra])
        
        ico14, ico_obj = _optimize_from_start(ico14, time_limit)
        ico_score = _true_score(ico14)
        if ico_score < best_score:
            best_score = ico_score
            best_pts = ico14.copy()
    
    # Final refinement with more iterations on the best result
    if best_pts is not None and time.time() - _t0 < time_limit:
        try:
            result = minimize(_objective, best_pts.flatten(), method='L-BFGS-B',
                            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10})
            refined_pts = result.x.reshape(N, D)
            refined_pts = refined_pts - refined_pts.mean(axis=0)
            refined_score = _true_score(refined_pts)
            if refined_score < best_score:
                best_score = refined_score
                best_pts = refined_pts.copy()
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: return Fibonacci sphere
        best_pts = _fib_sphere(N)
    
    return best_pts
