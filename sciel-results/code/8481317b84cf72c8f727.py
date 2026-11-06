import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # Safety margin from 150s limit

def _energy_and_grad(flat):
    """Compute logarithmic energy and gradient for flat array of 3N coordinates."""
    pts = flat.reshape(N, 3)
    
    # Normalize to sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    # Compute pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # Compute pairwise distances
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Avoid self-distances (diagonal)
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # We sum over all i != j and divide by 2, or just use upper triangle.
    # Using upper triangle is slightly faster and avoids the divide by 2.
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dists[iu])
    energy = -np.sum(log_dists)
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / d_ij^2
    # inv_dists_sq = 1 / d^2
    inv_dists_sq = 1.0 / (dists ** 2)
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    # grad_i = sum_j diff[i, j] * inv_dists_sq[i, j]
    # diff is (N, N, 3), inv_dists_sq is (N, N)
    # We need to multiply diff[i, j, k] by inv_dists_sq[i, j] and sum over j.
    grad = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad.flatten()

def _energy_only(flat):
    """Compute only the energy for optimization."""
    energy, _ = _energy_and_grad(flat)
    return energy

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on the unit sphere."""
    if seed is not None:
        np.random.seed(seed)
    vecs = np.random.randn(n, 3)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

def _optimize_from_start(start_pts):
    """Run L-BFGS-B optimization from a starting configuration."""
    x0 = start_pts.flatten()
    
    try:
        result = minimize(
            _energy_only,
            x0,
            jac=_energy_and_grad,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000}
        )
        
        if result.success or (hasattr(result, 'fun') and np.isfinite(result.fun)):
            best_x = result.x
            best_energy = result.fun
        else:
            best_x = x0
            best_energy = _energy_only(x0)
    except Exception:
        best_x = x0
        best_energy = _energy_only(x0)
    
    # Normalize the result
    best_pts = best_x.reshape(N, 3)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts, best_energy

def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    
    # Initial best
    best_pts = _fibonacci_sphere(N)
    best_energy = _energy_only(best_pts.flatten())
    
    # List of starting configurations
    starts = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=42),
        _random_sphere(N, seed=123),
        _random_sphere(N, seed=999)
    ]
    
    # Optimization phase 1: Try all starts
    for start in starts:
        if time.time() - start_time > DEADLINE - 20.0:
            break
            
        pts, energy = _optimize_from_start(start)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts
    
    # Optimization phase 2: Perturb and polish loop
    # If we have time, try to escape local minima by perturbing the best solution
    np.random.seed(42)
    while time.time() - start_time < DEADLINE - 5.0:
        # Perturb the best points
        noise_scale = 0.1
        perturbed = best_pts + noise_scale * np.random.randn(N, 3)
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / norms
        
        pts, energy = _optimize_from_start(perturbed)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts
            # Reduce noise scale as we get closer to optimum? Or keep it constant?
            # Constant is fine for now, L-BFGS will converge.
        else:
            # If we didn't improve, maybe try a smaller noise or different seed
            # For simplicity, just continue. The random seed changes each iteration.
            pass
    
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    # Validate: no points closer than 1e-9
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 1e-9:
        # Fall back to Fibonacci if invalid
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
