"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 100
TIME_LIMIT = 115.0  # Safety margin before 120s kill

# EVOLVE-BLOCK-START

def _log_energy(x):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)"""
    n = x.shape[0]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    energy = -np.sum(np.log(d))
    return energy

def _log_energy_grad(x):
    """Compute gradient of logarithmic energy with respect to x.
    
    Energy E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    """
    n = x.shape[0]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    dist_sq_safe = dist_sq.copy()
    np.fill_diagonal(dist_sq_safe, 1.0)
    
    coeff = 1.0 / dist_sq_safe  # (n, n)
    np.fill_diagonal(coeff, 0.0)
    
    # grad[i] = -sum_j coeff[i,j] * diff[i,j]
    grad = -np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return grad

def _project_to_sphere(x):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms

def _golden_spiral(n, seed_offset=0.0):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed_offset)
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere_points(n, rng):
    """Generate random points on the unit sphere."""
    v = rng.standard_normal((n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _optimize_from_start(x0, time_budget):
    """Run L-BFGS-B optimization from x0 with a time budget."""
    start_time = time.time()
    
    def objective_only(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _project_to_sphere(x)
        return _log_energy(x)
    
    x_flat = x0.flatten()
    
    best_x = x0.copy()
    best_e = _log_energy(x0)
    
    # Phase 1: Coarse optimization
    try:
        res = minimize(
            objective_only,
            x_flat,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        x_opt = _project_to_sphere(res.x.reshape(-1, 3))
        e_opt = _log_energy(x_opt)
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > time_budget * 0.7:
        return best_x, best_e
    
    # Phase 2: Fine optimization
    try:
        res = minimize(
            objective_only,
            best_x.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        x_opt = _project_to_sphere(res.x.reshape(-1, 3))
        e_opt = _log_energy(x_opt)
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
    except Exception:
        pass
    
    return best_x, best_e

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    
    # Golden spiral
    inits.append(_golden_spiral(N))
    
    # Perturbed golden spiral
    for scale in [0.1, 0.05]:
        perturbed = _golden_spiral(N) + scale * rng.standard_normal((N, 3))
        inits.append(_project_to_sphere(perturbed))
    
    # Random points
    for _ in range(3):
        inits.append(_random_sphere_points(N, rng))
    
    # Different golden spiral offsets
    for offset in [1.0, 2.0, 0.5]:
        inits.append(_golden_spiral(N, seed_offset=offset))
    
    best_x = None
    best_e = float('inf')
    
    for i, x0 in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = TIME_LIMIT - elapsed
        if remaining < 10:
            break
        
        # Allocate time budget for this optimization
        time_budget = max(5.0, remaining / max(1, len(inits) - i))
        
        x_opt, e_opt = _optimize_from_start(x0, time_budget)
        
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
        
        # Check if we're running low on time
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 5:
            break
    
    if best_x is None:
        best_x = _golden_spiral(N)
    
    # Final projection to ensure unit sphere
    best_x = _project_to_sphere(best_x)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < 1e-9:
                # Nudge point j slightly
                best_x[j] = best_x[j] + 1e-8 * rng.standard_normal(3)
                best_x[j] = _project_to_sphere(best_x[j])
    
    return best_x

# EVOLVE-BLOCK-END
