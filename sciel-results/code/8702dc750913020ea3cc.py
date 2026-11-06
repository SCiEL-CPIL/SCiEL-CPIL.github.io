"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


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
    # Compute pairwise squared distances
    sq_norms = np.sum(x * x, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (x @ x.T)
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    # Get upper triangle indices (i < j)
    iu = np.triu_indices(n, k=1)
    d_sq_upper = dist_sq[iu]
    # log(1/d) = -log(d) = -0.5 * log(d^2)
    energy = -0.5 * math.fsum(np.log(d_sq_upper).tolist())
    return energy


def _energy_and_grad(x_flat, time_limit, start_time):
    """Compute energy and projected gradient for the sphere-constrained problem."""
    if time.time() - start_time > time_limit - 5.0:
        raise TimeoutError()
    
    x = x_flat.reshape(N, 3)
    # Normalize to sphere
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms
    
    # Compute pairwise squared distances
    sq_norms = np.sum(x * x, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (x @ x.T)
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    mask = ~np.eye(N, dtype=bool)
    dist_sq_upper = dist_sq[mask]
    energy = -0.5 * np.sum(np.log(dist_sq_upper))
    
    # Gradient: grad_i = - sum_{j != i} (x_i - x_j) / dist_sq[i,j]
    W = np.zeros_like(dist_sq)
    W[mask] = 1.0 / dist_sq[mask]
    
    sum_W = W.sum(axis=1)
    grad_raw = -(x * sum_W[:, None] - (W @ x.T).T)
    
    # Project gradient onto tangent space
    dot_products = np.sum(grad_raw * x, axis=1)
    grad_proj = grad_raw - dot_products[:, None] * x
    
    return energy, grad_proj.flatten()


def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    # Start with Fibonacci sphere
    x_init = _fibonacci_sphere(N)
    best_x = x_init.copy()
    best_energy = _log_energy(best_x)
    
    # Run L-BFGS-B optimization
    try:
        result = minimize(
            _energy_and_grad,
            x_init.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-12, 'maxls': 50}
        )
        
        if result.success or result.fun < best_energy:
            x_opt = result.x.reshape(N, 3)
            # Normalize
            norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            x_opt = x_opt / norms
            
            e_opt = _log_energy(x_opt)
            if e_opt < best_energy:
                best_x = x_opt
                best_energy = e_opt
        
    except (TimeoutError, Exception):
        pass
    
    # If we have time, try a few more local optimizations from perturbed starts
    if time.time() - start_time < time_limit - 30.0:
        rng = np.random.default_rng(42)
        for restart in range(5):
            if time.time() - start_time > time_limit - 10.0:
                break
            
            # Create a perturbed version of the best solution
            noise = rng.normal(0, 0.1, size=(N, 3))
            x_perturbed = best_x + noise
            norms = np.linalg.norm(x_perturbed, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            x_perturbed = x_perturbed / norms
            
            try:
                result = minimize(
                    _energy_and_grad,
                    x_perturbed.flatten(),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-12, 'maxls': 50}
                )
                
                if result.success or result.fun < best_energy:
                    x_opt = result.x.reshape(N, 3)
                    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    x_opt = x_opt / norms
                    
                    e_opt = _log_energy(x_opt)
                    if e_opt < best_energy:
                        best_x = x_opt
                        best_energy = e_opt
                        
            except (TimeoutError, Exception):
                pass
    
    # Final validation
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_x = best_x / norms
    
    # Check for very close points
    dist_sq = np.sum((best_x[:, None, :] - best_x[None, :, :]) ** 2, axis=2)
    if np.min(dist_sq[np.triu_indices(N, k=1)]) < 1e-18:
        # Fall back to Fibonacci sphere if points are too close
        return _fibonacci_sphere(N)
    
    return best_x
# EVOLVE-BLOCK-END
