"""Logarithmic energy minimisation for N points on the unit sphere."""
import time
import numpy as np
import math
from scipy.optimize import minimize

N = 150
# Deadline guard: leave a safety margin (15 s) under the 150 s limit
_DEADLINE = time.time() + 135.0

def _energy(pts):
    """Vectorized sum_{i<j} log(1 / |x_i - x_j|) using math.fsum."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = dist[mask]
    d = d[d > 1e-15]
    if len(d) == 0:
        return 1e18
    terms = -np.log(d)
    return math.fsum(terms)

def _grad(pts):
    """Vectorized gradient of E wrt each point (tangent-projected)."""
    m = len(pts)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    inv_d2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_d2, 0.0)
    grad = np.einsum('ij,ijk->ik', inv_d2, diff)
    radial_comp = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - radial_comp * pts
    return grad

def _retract(pts):
    """Project points back to the sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 1, (n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _lbfgs_optimize(pts, max_time=30.0, tol=1e-10):
    """Optimize using L-BFGS-B on the sphere via tangent-space parameterization."""
    t_start = time.time()
    pts = pts.copy()
    E = _energy(pts)
    best_pts = pts.copy()
    best_E = E
    
    # Use a simple approach: optimize in 3D with projection after each step
    # L-BFGS-B doesn't handle manifold constraints well, so we use a 
    # penalty-free approach: optimize in ambient space and project
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        x = _retract(x)
        e = _energy(x)
        g = _grad(x)
        return e, g.flatten()
    
    x0 = pts.flatten()
    
    # Run L-BFGS-B with time-limited iterations
    result = minimize(
        lambda x: objective_and_grad(x)[0],
        x0,
        jac=lambda x: objective_and_grad(x)[1],
        method='L-BFGS-B',
        options={
            'maxiter': 5000,
            'ftol': tol,
            'gtol': tol,
            'maxfun': 5000,
            'maxls': 50,
            'maxcor': 20
        }
    )
    
    if result.success or result.fun < best_E:
        x_opt = _retract(result.x.reshape(N, 3))
        E_opt = _energy(x_opt)
        if E_opt < best_E:
            best_pts = x_opt
            best_E = E_opt
    
    # Check time
    if time.time() - t_start > max_time:
        pass  # Already stopped by maxiter
    
    return best_pts, best_E

def _polish(pts, max_time=15.0):
    """Final high-precision polish with very tight tolerances."""
    t_start = time.time()
    pts = pts.copy()
    E = _energy(pts)
    best_pts = pts.copy()
    best_E = E
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        x = _retract(x)
        e = _energy(x)
        g = _grad(x)
        return e, g.flatten()
    
    x0 = pts.flatten()
    
    result = minimize(
        lambda x: objective_and_grad(x)[0],
        x0,
        jac=lambda x: objective_and_grad(x)[1],
        method='L-BFGS-B',
        options={
            'maxiter': 10000,
            'ftol': 1e-14,
            'gtol': 1e-14,
            'maxfun': 10000,
            'maxls': 50,
            'maxcor': 30
        }
    )
    
    if result.fun < best_E:
        x_opt = _retract(result.x.reshape(N, 3))
        E_opt = _energy(x_opt)
        if E_opt < best_E:
            best_pts = x_opt
            best_E = E_opt
    
    return best_pts, best_E

def construct():
    t0 = time.time()
    
    # 1. Start with Fibonacci spiral
    pts0 = _fibonacci_sphere(N)
    best_pts, best_E = _lbfgs_optimize(pts0, max_time=20.0)
    
    # 2. Multiple diverse restarts
    seeds = [42, 123, 999, 2023, 777, 555, 314, 271, 161, 83]
    for seed in seeds:
        if time.time() > _DEADLINE - 30:
            break
        p = _random_sphere(N, seed)
        p2, E2 = _lbfgs_optimize(p, max_time=15.0)
        if E2 < best_E:
            best_pts, best_E = p2, E2
    
    # 3. Jittered Fibonacci for more diversity
    if time.time() < _DEADLINE - 25:
        rng = np.random.default_rng(12345)
        for _ in range(3):
            if time.time() > _DEADLINE - 20:
                break
            p = _fibonacci_sphere(N)
            p += rng.normal(0, 0.1, (N, 3))
            p = _retract(p)
            p2, E2 = _lbfgs_optimize(p, max_time=15.0)
            if E2 < best_E:
                best_pts, best_E = p2, E2
    
    # 4. Final high-precision polish
    if time.time() < _DEADLINE - 10:
        polished_pts, polished_E = _polish(best_pts, max_time=10.0)
        if polished_E < best_E:
            best_pts, best_E = polished_pts, polished_E

    # Final normalization
    best_pts = _retract(best_pts)
    
    return best_pts
