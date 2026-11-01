import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 150.0
DEADLINE_FRACTION = 0.88

def _log_energy_fast(pts):
    """Vectorized log energy. Uses math.fsum for precision if needed, but np.sum is usually fine for relative comparison.
    The task specifies math.fsum for the score, but for optimization, np.sum is faster.
    We will use the exact formula for the final check, but for the loop, vectorized is key."""
    n = pts.shape[0]
    # Compute squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    
    # To avoid log(0), we assume points are distinct. 
    # For optimization stability, we clip very small distances.
    # However, the true energy is -inf if d=0. 
    # We want to minimize, so we want to avoid d=0.
    # We'll use a small epsilon to prevent numerical issues in the log during optimization.
    eps = 1e-15
    dists2 = np.maximum(dists2, eps)
    
    # Energy is sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij^2) / 2 ? 
    # No, sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij).
    # d_ij = sqrt(d_ij^2). log(d_ij) = 0.5 log(d_ij^2).
    # So sum_{i<j} -0.5 log(d_ij^2).
    
    iu = np.triu_indices(n, k=1)
    # log(1/d) = -log(d) = -0.5 log(d^2)
    energy = -0.5 * np.sum(np.log(dists2[iu]))
    
    return energy

def _log_energy_grad_fast(pts):
    """Vectorized log energy and gradient."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    
    eps = 1e-15
    dists2 = np.maximum(dists2, eps)
    
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dists2[iu]))
    
    # Gradient of E = sum_{i<j} log(1/|xi-xj|)
    # dE/dxi = -sum_{j!=i} (xi-xj) / |xi-xj|^2
    # Note: The term (xi-xj)/|xi-xj|^2 is the vector from xj to xi divided by squared distance.
    
    inv_dists2 = 1.0 / dists2
    np.fill_diagonal(inv_dists2, 0.0)
    
    # s_i = sum_j (xi-xj)/d_ij^2
    s = (diff * inv_dists2[:, :, np.newaxis]).sum(axis=1)
    
    # grad = -s
    grad = -s
    
    return energy, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _tangent_project(pts, grad):
    """Project gradient to tangent space of sphere."""
    coeffs = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - coeffs * pts

def _fibonacci_spiral(n=N):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n=N, seed=None):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)

def _optimize_lbfgs(pts, deadline):
    """Run L-BFGS-B optimization with projected gradients."""
    pts = _project_to_sphere(pts)
    x0 = pts.flatten()
    
    def func(x):
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _log_energy_grad_fast(p)
        return e
        
    def jac(x):
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _log_energy_grad_fast(p)
        g_t = _tangent_project(p, g)
        return g_t.flatten()
        
    options = {
        'maxiter': 2000,
        'gtol': 1e-12,
        'ftol': 1e-15,
        'disp': False,
        'maxfun': 10000,
    }
    
    try:
        result = minimize(func, x0, jac=jac, method='L-BFGS-B', options=options)
        if result.fun < 1e17:
            return _project_to_sphere(result.x.reshape(-1, 3))
    except Exception:
        pass
        
    return pts

def _polish(pts, deadline, lr_init=0.05, min_lr=1e-16):
    """Steepest descent polish with backtracking line search."""
    pts = _project_to_sphere(pts)
    current_energy = _log_energy_fast(pts)
    
    lr = lr_init
    patience = 0
    
    while time.time() < deadline:
        e, g = _log_energy_grad_fast(pts)
        if e < -1e17: break
        
        g_t = _tangent_project(pts, g)
        grad_norm = np.linalg.norm(g_t)
        if grad_norm < 1e-14: break
        
        alpha = lr
        found = False
        for _ in range(50):
            new_pts = pts - alpha * g_t
            new_pts = _project_to_sphere(new_pts)
            
            # Check for collisions (optional but good for stability)
            # If any distance is too small, reject step? 
            # The energy function handles this with eps, but let's be safe.
            
            new_e = _log_energy_fast(new_pts)
            
            # Armijo condition
            c = 1e-4
            if new_e < current_energy + c * alpha * np.sum(g_t * g_t):
                pts = new_pts
                current_energy = new_e
                found = True
                break
            alpha *= 0.5
        
        if found:
            patience = 0
            lr = min(alpha * 1.5, 1.0)
        else:
            lr = max(alpha * 0.5, min_lr)
            patience += 1
            if patience > 20 or lr < min_lr:
                break
                
    return pts

def construct():
    global _start_time
    _start_time = time.time()
    deadline = _start_time + TIME_LIMIT * DEADLINE_FRACTION
    
    best_pts = None
    best_energy = float('inf')
    
    # Generate diverse starts
    rng = np.random.default_rng(12345)
    fib = _fibonacci_spiral(N)
    
    starts = [
        fib,
        _project_to_sphere(fib + rng.normal(0, 0.05, (N, 3))),
        _project_to_sphere(fib + rng.normal(0, 0.15, (N, 3))),
        _project_to_sphere(fib + rng.normal(0, 0.3, (N, 3))),
        _random_sphere(N, seed=1),
        _random_sphere(N, seed=2),
        _random_sphere(N, seed=3),
    ]
    
    # Filter similar starts
    unique_starts = [starts[0]]
    for s in starts[1:]:
        if all(np.linalg.norm(s - u) > 0.2 for u in unique_starts):
            unique_starts.append(s)
    starts = unique_starts
    
    total_time = deadline - _start_time
    opt_deadline = _start_time + total_time * 0.7
    polish_deadline = deadline
    
    for i, start in enumerate(starts):
        if time.time() > opt_deadline:
            break
            
        optimized = _optimize_lbfgs(start, opt_deadline)
        e = _log_energy_fast(optimized)
        
        if e < best_energy:
            best_energy = e
            best_pts = optimized
            
    if best_pts is not None:
        # Polish the best result
        best_pts = _polish(best_pts, polish_deadline - 5.0)
        
        # Final high precision polish with very small steps
        if time.time() < polish_deadline - 2.0:
            best_pts = _polish(best_pts, polish_deadline, lr_init=1e-4, min_lr=1e-18)
            
        best_pts = _project_to_sphere(best_pts)
        
        # Validity check
        if not np.all(np.isfinite(best_pts)):
            return _fibonacci_spiral(N)
            
        diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        if min_d < 1e-9:
            return _fibonacci_spiral(N)
            
    else:
        best_pts = _fibonacci_spiral(N)
        
    return best_pts
