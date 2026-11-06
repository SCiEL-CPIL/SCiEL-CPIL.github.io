"""Optimized program for logarithmic energy minimization on the sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # Safety margin from 150s limit


def _energy_and_grad(flat):
    """Compute logarithmic energy and gradient for flat array of 3N coordinates."""
    pts = flat.reshape(N, 3)
    # Normalize to sphere (in case of numerical drift)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Avoid self-distances and very small distances
    np.fill_diagonal(dists, 1.0)
    dists = np.maximum(dists, 1e-12)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dists[iu])
    energy = -np.sum(log_dists)
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / d_ij^2
    # For the full energy, grad_i = sum_{j!=i} (x_i - x_j) / d_ij^2
    grad = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = dists[i, j]
            grad[i] += (pts[i] - pts[j]) / (d * d)
    
    return energy, grad.flatten()


def _energy_only(flat):
    """Compute only the energy for optimization."""
    pts = flat.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    dists = np.maximum(dists, 1e-12)
    
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dists[iu])
    return -np.sum(log_dists)


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize_from_start(start_pts, max_time):
    """Run L-BFGS-B optimization from a starting configuration."""
    x0 = start_pts.flatten()
    
    best_energy = _energy_only(x0)
    best_x = x0.copy()
    
    start_time = time.time()
    
    # Use L-BFGS-B with numerical gradient (or provide analytic gradient)
    # We'll use our analytic gradient for efficiency
    try:
        result = minimize(
            _energy_only,
            x0,
            jac=_energy_and_grad,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
        )
        
        if result.success or result.fun < best_energy:
            best_x = result.x
            best_energy = result.fun
    except Exception:
        pass
    
    # Normalize the result
    best_pts = best_x.reshape(N, 3)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts, best_energy


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    
    # Start with Fibonacci spiral
    start_pts = _fibonacci_sphere(N)
    
    # Run optimization
    pts, energy = _optimize_from_start(start_pts, DEADLINE - (time.time() - start_time))
    
    # Check deadline - if we're close, return what we have
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5.0:
        # Normalize and return
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts = pts / norms
        return pts
    
    # Try a second restart with perturbed initial points if time allows
    if time.time() - start_time < DEADLINE - 30.0:
        np.random.seed(42)
        perturbed = start_pts + 0.01 * np.random.randn(N, 3)
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / norms
        
        pts2, energy2 = _optimize_from_start(perturbed, DEADLINE - (time.time() - start_time))
        
        if energy2 < energy:
            pts = pts2
            energy = energy2
    
    # Final normalization
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    # Validate: no points closer than 1e-9
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 1e-9:
        # Fall back to Fibonacci if invalid
        pts = _fibonacci_sphere(N)
    
    return pts
