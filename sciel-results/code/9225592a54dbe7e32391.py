"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize


def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    # points: (N, 3) array
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Mask diagonal and upper triangle
    iu = np.triu_indices(N, 1)
    d = dist[iu]
    # Avoid log(0) or negative values
    d = np.maximum(d, 1e-15)
    return math.fsum(math.log(1.0 / di) for di in d)


def _log_energy_grad(points):
    """Gradient of logarithmic energy w.r.t. points on sphere (projected)."""
    # points: (N, 3)
    # E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|x_i - x_j|
    # dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # (each pair (i,j) contributes (x_i-x_j)/|x_i-x_j|^2 to grad_i and (x_j-x_i)/|x_j-x_i|^2 to grad_j)
    
    grad = np.zeros_like(points)
    for i in range(N):
        for j in range(i + 1, N):
            diff = points[i] - points[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-20:
                continue
            inv_d2 = 1.0 / d2
            grad[i] += diff * inv_d2
            grad[j] -= diff * inv_d2
    
    # Project gradient to tangent space of sphere
    # For each point, subtract component along the point itself
    for i in range(N):
        grad[i] -= np.dot(grad[i], points[i]) * points[i]
    
    return grad


def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_spiral(perturb=0.0, seed=None):
    """Generate Fibonacci spiral points on the unit sphere with optional perturbation."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    
    if perturb > 0:
        phi += rng.uniform(-perturb, perturb, N)
    
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    if perturb > 0:
        pts += rng.normal(0, perturb * 0.1, size=(N, 3))
        pts = _project_to_sphere(pts)
    
    return pts


def construct():
    """
    Use a multi-start optimization approach with correct gradient:
    1. Generate multiple initial configurations (Fibonacci spiral with perturbations)
    2. For each, run L-BFGS-B optimization with sphere constraint via projection
    3. Return the best configuration found
    """
    best_points = None
    best_energy = float('inf')
    
    # Strategy: Use L-BFGS with gradient projection onto tangent space
    # Initialize with Fibonacci spiral + random perturbations
    
    np.random.seed(42)
    
    # Multiple trials with different perturbations
    for trial in range(8):
        if trial == 0:
            x0 = _fibonacci_spiral(perturb=0.0)
        elif trial == 1:
            x0 = _fibonacci_spiral(perturb=0.05, seed=trial)
        elif trial == 2:
            x0 = _fibonacci_spiral(perturb=0.1, seed=trial)
        elif trial == 3:
            x0 = _fibonacci_spiral(perturb=0.2, seed=trial)
        else:
            x0 = _fibonacci_spiral(perturb=0.3 + 0.1 * trial, seed=trial)
        
        # Flatten for optimizer
        x_flat = x0.flatten()
        
        def objective_and_grad(x_flat, _N=N):
            pts = x_flat.reshape(_N, 3)
            # Project to sphere
            pts = _project_to_sphere(pts)
            e = _log_energy(pts)
            g = _log_energy_grad(pts)
            return e, g.flatten()
        
        # Use L-BFGS-B
        try:
            res = minimize(
                objective_and_grad,
                x_flat,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12, 'maxfun': 10000}
            )
            pts = _project_to_sphere(res.x.reshape(N, 3))
            e = _log_energy(pts)
            
            if e < best_energy:
                best_energy = e
                best_points = pts
        except Exception:
            continue
    
    # If no optimization succeeded, fall back to Fibonacci spiral
    if best_points is None:
        best_points = _fibonacci_spiral(perturb=0.0)
    
    return best_points
# EVOLVE-BLOCK-END
