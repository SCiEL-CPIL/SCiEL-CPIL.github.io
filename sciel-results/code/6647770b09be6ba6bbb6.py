import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

def _normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

def _log_energy_and_grad(x_flat, start_time, time_limit):
    """Compute log energy and projected gradient."""
    if time.time() - start_time > time_limit - 2.0:
        raise TimeoutError()
    
    x = x_flat.reshape(N, 3)
    x = _normalize(x)
    
    # Compute squared distances
    sq_norms = np.sum(x * x, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (x @ x.T)
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    mask = ~np.eye(N, dtype=bool)
    d_sq_upper = dist_sq[mask]
    
    # Energy: sum_{i<j} log(1/d) = -0.5 * sum_{i!=j} log(d) = -0.5 * sum log(d_sq) / 2 ?
    # log(1/d) = -log(d) = -0.5 log(d^2).
    # Sum over i<j is half of sum over i!=j.
    # So E = -0.5 * sum_{i!=j} (-0.5 log(d_ij^2)) ? No.
    # E = sum_{i<j} -log(d_ij) = -0.5 * sum_{i!=j} log(d_ij).
    # log(d_ij) = 0.5 log(d_ij^2).
    # E = -0.5 * sum_{i!=j} 0.5 log(d_ij^2) = -0.25 * sum_{i!=j} log(d_ij^2).
    
    energy = -0.25 * np.sum(np.log(d_sq_upper))
    
    # Gradient
    # grad_i = sum_{j != i} (x_i - x_j) / d_ij^2
    # W_ij = 1 / d_ij^2
    W = np.zeros_like(dist_sq)
    W[mask] = 1.0 / d_sq_upper
    
    sum_W = W.sum(axis=1)
    grad_raw = x * sum_W[:, None] - (W @ x.T).T
    
    # Project gradient to tangent space
    dot_products = np.sum(grad_raw * x, axis=1)
    grad_proj = grad_raw - dot_products[:, None] * x
    
    return energy, grad_proj.flatten()

def _coulomb_energy_and_grad(x_flat, start_time, time_limit):
    """Compute Coulomb energy and projected gradient."""
    if time.time() - start_time > time_limit - 2.0:
        raise TimeoutError()
    
    x = x_flat.reshape(N, 3)
    x = _normalize(x)
    
    sq_norms = np.sum(x * x, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (x @ x.T)
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    mask = ~np.eye(N, dtype=bool)
    d_sq_upper = dist_sq[mask]
    
    # E = sum_{i<j} 1/d_ij = 0.5 * sum_{i!=j} 1/sqrt(d_ij^2)
    d_upper = np.sqrt(d_sq_upper)
    energy = 0.5 * np.sum(1.0 / d_upper)
    
    # grad_i = -sum_{j != i} (x_i - x_j) / d_ij^3
    # W_ij = 1 / d_ij^3
    W = np.zeros_like(dist_sq)
    W[mask] = 1.0 / (d_upper * dist_sq[mask]) # 1/d * 1/d^2 = 1/d^3
    
    sum_W = W.sum(axis=1)
    grad_raw = x * sum_W[:, None] - (W @ x.T).T
    
    dot_products = np.sum(grad_raw * x, axis=1)
    grad_proj = grad_raw - dot_products[:, None] * x
    
    return energy, grad_proj.flatten()

def _optimize(x_start, start_time, time_limit, energy_func, maxiter=2000, tol=1e-10):
    """Run optimization and return best x."""
    try:
        result = minimize(
            energy_func,
            x_start.flatten(),
            method='L-BFGS-B',
            jac=True,
            args=(start_time, time_limit),
            options={'maxiter': maxiter, 'gtol': tol, 'ftol': 1e-12, 'maxls': 50}
        )
        
        if result.fun < np.inf:
            x_opt = result.x.reshape(N, 3)
            x_opt = _normalize(x_opt)
            return x_opt
    except (TimeoutError, Exception):
        pass
    return None

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # Initial candidates
    x_fib = _fibonacci_sphere(N)
    rng = np.random.default_rng(42)
    x_rand = _normalize(rng.normal(0, 1, size=(N, 3)))
    
    candidates = [x_fib, x_rand]
    
    # Try to improve initial candidates with Coulomb optimization
    # This is a heuristic: Coulomb minimizers are often good Log minimizers
    if time.time() - start_time < time_limit - 20.0:
        for x_init in list(candidates):
            x_coulomb = _optimize(x_init, start_time, time.time() - start_time + 10.0, _coulomb_energy_and_grad, maxiter=1000, tol=1e-8)
            if x_coulomb is not None:
                candidates.append(x_coulomb)
    
    best_x = None
    best_energy = np.inf
    
    # First pass: Optimize Log energy from all candidates
    for x_start in candidates:
        if time.time() - start_time > time_limit - 15.0:
            break
        
        x_opt = _optimize(x_start, start_time, time_limit, _log_energy_and_grad, maxiter=2000, tol=1e-10)
        if x_opt is not None:
            # Calculate energy accurately
            e = _log_energy_and_grad(x_opt.flatten(), start_time, time_limit)[0]
            if e < best_energy:
                best_energy = e
                best_x = x_opt
    
    # Second pass: Perturb the best solution and optimize again
    if best_x is not None and time.time() - start_time < time_limit - 10.0:
        for scale in [0.1, 0.2, 0.5, 1.0]:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            noise = rng.normal(0, scale, size=(N, 3))
            x_perturbed = _normalize(best_x + noise)
            
            x_opt = _optimize(x_perturbed, start_time, time_limit, _log_energy_and_grad, maxiter=2000, tol=1e-10)
            if x_opt is not None:
                e = _log_energy_and_grad(x_opt.flatten(), start_time, time_limit)[0]
                if e < best_energy:
                    best_energy = e
                    best_x = x_opt
    
    # Final polish: Tighter tolerance on the best solution
    if best_x is not None and time.time() - start_time < time_limit - 2.0:
        x_polished = _optimize(best_x, start_time, time_limit, _log_energy_and_grad, maxiter=5000, tol=1e-14)
        if x_polished is not None:
            e = _log_energy_and_grad(x_polished.flatten(), start_time, time_limit)[0]
            if e < best_energy:
                best_energy = e
                best_x = x_polished
    
    if best_x is None:
        return x_fib
        
    # Final validation
    dist_sq = np.sum((best_x[:, None, :] - best_x[None, :, :]) ** 2, axis=2)
    if np.min(dist_sq[np.triu_indices(N, k=1)]) < 1e-18:
        return _fibonacci_sphere(N)
        
    return best_x
