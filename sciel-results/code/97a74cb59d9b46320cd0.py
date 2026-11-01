import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 150.0
DEADLINE_FRACTION = 0.88

def _log_energy_fast(pts):
    """Vectorized log energy."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists2, 1.0)
    iu = np.triu_indices(n, k=1)
    # log(1/d) = -0.5 * log(d^2)
    return -0.5 * np.sum(np.log(dists2[iu]))

def _log_energy_grad_fast(pts):
    """Vectorized log energy and gradient (Euclidean, not projected)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists2, 1.0)
    
    # Energy: -0.5 * sum log(d^2) over upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dists2[iu]))
    
    # Gradient of E = sum_{i<j} log(1/|xi-xj|) = -0.5 * sum_{i<j} log(|xi-xj|^2)
    # dE/dxi = -0.5 * sum_{j!=i} d/dxi log(|xi-xj|^2)
    #        = -0.5 * sum_{j!=i} 2*(xi-xj)/|xi-xj|^2
    #        = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    # Wait, let me recompute. E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dxi = -sum_{j>i} d/dxi log(d_ij) - sum_{j<i} d/dxi log(d_ji)
    # d/dxi log(d_ij) = (xi-xj)/d_ij^2
    # So dE/dxi = -sum_{j!=i} (xi-xj)/d_ij^2
    # But the code has grad = sum_j (xi-xj)/d^2, which is the negative of what we want for minimization.
    # Actually for L-BFGS-B we provide the gradient of the objective to minimize.
    # The objective is E, so we need dE/dxi = -sum_{j!=i} (xi-xj)/d_ij^2
    
    inv_dists2 = 1.0 / dists2
    np.fill_diagonal(inv_dists2, 0.0)
    
    # sum_j (xi-xj)/d_ij^2
    s = (diff * inv_dists2[:, :, np.newaxis]).sum(axis=1)
    
    # Gradient of E is -s
    grad = -s
    
    return energy, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _tangent_project(pts, grad):
    """Project gradient to tangent space of sphere."""
    # Tangent space at x: {v : x·v = 0}
    # Projection: v_t = v - (v·x)x
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

def _icosahedron_based(n=N):
    """Start from icosahedron vertices and subdivide."""
    # Icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _project_to_sphere(verts)
    
    # Simple approach: just use random points but seeded differently
    return _random_sphere(n, seed=42)

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
        # Project gradient to tangent space for better conditioning
        g_t = _tangent_project(p, g)
        return g_t.flatten()
        
    options = {
        'maxiter': 3000,
        'gtol': 1e-14,
        'ftol': 1e-18,
        'disp': False,
        'maxfun': 15000,
    }
    
    try:
        result = minimize(func, x0, jac=jac, method='L-BFGS-B', options=options)
        if result.fun < 1e17:
            return _project_to_sphere(result.x.reshape(-1, 3))
    except Exception:
        pass
        
    return pts

def _riemannian_cg(pts, deadline):
    """Riemannian Conjugate Gradient optimization."""
    pts = _project_to_sphere(pts)
    current_energy = _log_energy_fast(pts)
    
    e, g = _log_energy_grad_fast(pts)
    g_t = _tangent_project(pts, g)
    d = -g_t
    d = _tangent_project(pts, d)
    
    for k in range(500):
        if time.time() > deadline:
            break
            
        # Line search
        alpha = 0.01
        found = False
        for _ in range(50):
            v = alpha * d
            new_pts = pts + v
            new_pts = _project_to_sphere(new_pts)
            new_e = _log_energy_fast(new_pts)
            
            # Armijo condition
            if new_e < current_energy + 1e-4 * alpha * np.sum(g_t * d):
                pts = new_pts
                current_energy = new_e
                found = True
                break
            alpha *= 0.5
            
        if not found:
            break
            
        # Compute new gradient
        e_new, g_new = _log_energy_grad_fast(pts)
        g_t_new = _tangent_project(pts, g_new)
        
        if np.linalg.norm(g_t_new) < 1e-15:
            break
            
        # Vector transport (approximate parallel transport via projection)
        d_trans = d - np.sum(d * pts, axis=1, keepdims=True) * pts
        
        # Polak-Ribiere beta
        denom = np.sum(g_t * g_t)
        if denom < 1e-15:
            beta = 0.0
        else:
            beta = np.sum(g_t_new * (g_t_new - g_t)) / denom
            if beta < 0:
                beta = 0.0
                
        d_new = -g_t_new + beta * d_trans
        d_new = _tangent_project(pts, d_new)
        
        # Restart if not descent
        if np.sum(d_new * g_t_new) > 0:
            d_new = -g_t_new
            d_new = _tangent_project(pts, d_new)
            
        d = d_new
        g_t = g_t_new
        
    return pts

def construct():
    global _start_time
    _start_time = time.time()
    deadline = _start_time + TIME_LIMIT * DEADLINE_FRACTION
    
    best_pts = None
    best_energy = float('inf')
    
    rng = np.random.default_rng(12345)
    fib = _fibonacci_spiral(N)
    starts = [
        fib,
        _project_to_sphere(fib + rng.normal(0, 0.05, (N, 3))),
        _project_to_sphere(fib + rng.normal(0, 0.15, (N, 3))),
        _random_sphere(N, seed=1),
        _random_sphere(N, seed=7),
        _random_sphere(N, seed=13),
        _random_sphere(N, seed=21),
    ]
    
    # Allocate time
    total_time = deadline - _start_time
    opt_deadline = _start_time + total_time * 0.7
    polish_deadline = deadline
    
    for i, start in enumerate(starts):
        if time.time() > opt_deadline:
            break
            
        # 1. Coarse optimization with L-BFGS-B
        optimized = _optimize_lbfgs(start, opt_deadline)
        
        # 2. Fine optimization with Riemannian CG
        polished = _riemannian_cg(optimized, polish_deadline - 5.0)
        
        e = _log_energy_fast(polished)
        
        if e < best_energy:
            best_energy = e
            best_pts = polished
            
    if best_pts is not None:
        # Final very tight polish
        best_pts = _riemannian_cg(best_pts, deadline)
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
