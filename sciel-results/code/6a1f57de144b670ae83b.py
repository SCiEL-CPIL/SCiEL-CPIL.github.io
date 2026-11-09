"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy_and_grad(points):
    """Compute log energy and its gradient for points on the unit sphere.
    Energy = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    Gradient wrt xi: -sum_{j!=i} (xi - xj)/|xi - xj|^2
    """
    n = points.shape[0]
    # Pairwise distances
    # diff[i,j] = xi - xj
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    log_dist = np.log(dist[iu])
    energy = -np.sum(log_dist)
    
    # Gradient: for each point i, grad_i = -sum_{j!=i} (xi - xj) / |xi-xj|^2
    # = -sum_{j!=i} diff[i,j] / dist_sq[i,j]
    # But we need to be careful: the constraint is that points are on the sphere.
    # We'll project the gradient onto the tangent space later, or use a parametrization.
    
    # Actually, for optimization on the sphere, we can use the fact that
    # the gradient in R^3 projected onto the tangent space is what matters.
    # grad_i (unconstrained) = -sum_{j!=i} (xi - xj)/|xi-xj|^2
    # Projected: grad_i - (grad_i . xi) xi
    
    grad_unconstrained = -np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)  # (n, 3)
    
    # Project onto tangent space
    dot = np.sum(grad_unconstrained * points, axis=1, keepdims=True)
    grad = grad_unconstrained - dot * points
    
    return energy, grad


def _energy_only(points):
    """Compute just the energy."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(rng):
    """Generate random points on the unit sphere."""
    v = rng.normal(size=(N, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def _jittered_fibonacci(rng, sigma=0.05):
    """Fibonacci spiral with small random jitter, then renormalize."""
    base = _fibonacci_spiral()
    noise = rng.normal(0, sigma, size=(N, 3))
    pts = base + noise
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _optimize_from_start(start_points, deadline):
    """Run gradient-based optimization from a starting point, with time guard."""
    pts = start_points.copy()
    best_energy = _energy_only(pts)
    best_pts = pts.copy()
    
    t_start = time.time()
    max_iter = 5000
    
    # Use L-BFGS-B with a projection-based approach
    # We'll use a simple approach: optimize in R^3 and project back
    
    def objective_and_grad(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(pts)
        # The gradient in the ambient space needs to account for the projection
        # For simplicity, use the projected gradient as the gradient of the objective
        # This is an approximation but works well in practice
        return e, g.reshape(-1)
    
    # Use L-BFGS-B
    try:
        result = minimize(
            objective_and_grad,
            pts.reshape(-1),
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'gtol': 1e-10,
                'ftol': 1e-12,
                'maxls': 50,
                'maxfun': max_iter * 10,
            }
        )
        if result.success or result.fun < best_energy:
            best_pts = _project_to_sphere(result.x.reshape(N, 3))
            best_energy = _energy_only(best_pts)
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_pts, best_energy
    
    # Second pass with tighter tolerance
    t_start = time.time()
    max_iter2 = 3000
    try:
        result2 = minimize(
            objective_and_grad,
            best_pts.reshape(-1),
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter2,
                'gtol': 1e-12,
                'ftol': 1e-14,
                'maxls': 100,
                'maxfun': max_iter2 * 10,
            }
        )
        if result2.fun < best_energy:
            best_pts = _project_to_sphere(result2.x.reshape(N, 3))
            best_energy = _energy_only(best_pts)
    except Exception:
        pass
    
    return best_pts, best_energy


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    t0 = time.time()
    deadline = t0 + 140.0  # Leave 10s safety margin
    
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_spiral())
    
    # 2. Random points
    starts.append(_random_points(rng))
    
    # 3. Jittered Fibonacci
    starts.append(_jittered_fibonacci(rng, sigma=0.05))
    
    # 4. Another random
    starts.append(_random_points(rng))
    
    # 5. Jittered Fibonacci with larger sigma
    starts.append(_jittered_fibonacci(rng, sigma=0.1))
    
    # 6. Another random with different seed
    rng2 = np.random.RandomState(123)
    starts.append(_random_points(rng2))
    
    best_pts = starts[0].copy()
    best_energy = _energy_only(best_pts)
    
    # Try each starting point
    for i, start in enumerate(starts):
        if time.time() > deadline:
            break
        
        pts, energy = _optimize_from_start(start, deadline)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # If we have time, try another optimization from the best found so far
        if time.time() < deadline - 20.0:
            pts2, energy2 = _optimize_from_start(best_pts, deadline)
            if energy2 < best_energy:
                best_energy = energy2
                best_pts = pts2.copy()
    
    # Final check: ensure points are valid
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge points apart
                best_pts[j] += 1e-8 * rng.normal(size=3)
                best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
