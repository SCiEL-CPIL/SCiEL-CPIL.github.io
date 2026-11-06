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

def _log_energy(x):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = x.shape[0]
    sq_norms = np.sum(x * x, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (x @ x.T)
    dist_sq = np.maximum(dist_sq, 1e-15)
    iu = np.triu_indices(n, k=1)
    d_sq_upper = dist_sq[iu]
    energy = -0.5 * math.fsum(np.log(d_sq_upper).tolist())
    return energy

def _energy_and_grad(x_flat, start_time, time_limit):
    """Compute energy and projected gradient for the sphere-constrained problem."""
    if time.time() - start_time > time_limit - 5.0:
        raise TimeoutError()
    
    x = x_flat.reshape(N, 3)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms
    
    sq_norms = np.sum(x * x, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (x @ x.T)
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    mask = ~np.eye(N, dtype=bool)
    dist_sq_upper = dist_sq[mask]
    
    energy = -0.5 * np.sum(np.log(dist_sq_upper))
    
    W = np.zeros_like(dist_sq)
    W[mask] = 1.0 / dist_sq[mask]
    
    sum_W = W.sum(axis=1)
    grad_raw = -(x * sum_W[:, None] - (W @ x.T).T)
    
    dot_products = np.sum(grad_raw * x, axis=1)
    grad_proj = grad_raw - dot_products[:, None] * x
    
    return energy, grad_proj.flatten()

def _optimize_from(x_start, start_time, time_limit, maxiter=2000):
    """Run optimization from a starting point and return the best result."""
    try:
        result = minimize(
            _energy_and_grad,
            x_start.flatten(),
            method='L-BFGS-B',
            jac=True,
            args=(start_time, time_limit),
            options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-12, 'maxls': 50}
        )
        
        if result.fun < np.inf:
            x_opt = result.x.reshape(N, 3)
            norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            x_opt = x_opt / norms
            return x_opt
    except (TimeoutError, Exception):
        pass
    return None

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # Start with Fibonacci sphere
    x_init = _fibonacci_sphere(N)
    best_x = x_init.copy()
    best_energy = _log_energy(best_x)
    
    # Generate diverse starting points
    rng = np.random.default_rng(42)
    starts = [x_init]
    
    # Add some random starts with different scales
    for scale in [0.05, 0.1, 0.2, 0.3, 0.5]:
        noise = rng.normal(0, scale, size=(N, 3))
        x_perturbed = best_x + noise
        norms = np.linalg.norm(x_perturbed, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        x_perturbed = x_perturbed / norms
        starts.append(x_perturbed)
    
    # Also add a few fully random starts
    for _ in range(3):
        x_rand = rng.normal(0, 1, size=(N, 3))
        norms = np.linalg.norm(x_rand, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        x_rand = x_rand / norms
        starts.append(x_rand)
    
    # Optimize from each starting point
    for x_start in starts:
        if time.time() - start_time > time_limit - 10.0:
            break
        
        result = _optimize_from(x_start, start_time, time_limit, maxiter=2000)
        if result is not None:
            e = _log_energy(result)
            if e < best_energy:
                best_x = result
                best_energy = e
    
    # Second round: more aggressive restarts from the best solution
    if time.time() - start_time < time_limit - 20.0:
        for scale in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]:
            if time.time() - start_time > time_limit - 10.0:
                break
            
            noise = rng.normal(0, scale, size=(N, 3))
            x_perturbed = best_x + noise
            norms = np.linalg.norm(x_perturbed, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            x_perturbed = x_perturbed / norms
            
            result = _optimize_from(x_perturbed, start_time, time_limit, maxiter=1500)
            if result is not None:
                e = _log_energy(result)
                if e < best_energy:
                    best_x = result
                    best_energy = e
    
    # Final validation
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_x = best_x / norms
    
    dist_sq = np.sum((best_x[:, None, :] - best_x[None, :, :]) ** 2, axis=2)
    if np.min(dist_sq[np.triu_indices(N, k=1)]) < 1e-18:
        return _fibonacci_sphere(N)
    
    return best_x
