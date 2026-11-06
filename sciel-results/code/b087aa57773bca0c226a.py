"""Optimize 100 points on the unit sphere to minimize logarithmic energy."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 110.0  # safety margin below 120s

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using golden angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient for points on sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient: for each point, sum over other points of (x_i - x_j)/|x_i - x_j|^2
    """
    n = len(pts)
    # Compute pairwise distances
    # dists[i,j] = |x_i - x_j|
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Avoid division by zero (diagonal)
    mask = ~np.eye(n, dtype=bool)
    
    # Energy: -sum_{i<j} log(dists[i,j])
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient: for point i, grad_i = sum_{j!=i} (x_i - x_j) / dists[i,j]^2
    # Note: d/dx_i [-log|x_i - x_j|] = -(x_i - x_j)/|x_i - x_j|^2
    # So grad_i = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    # Wait, let me be careful:
    # E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    # dE/dx_i for fixed i: sum_{j!=i} d/dx_i [-log|x_i - x_j|]
    # For j < i: term is -log|x_i - x_j|, d/dx_i = -(x_i - x_j)/|x_i - x_j|^2
    # For j > i: term is -log|x_j - x_i| = -log|x_i - x_j|, d/dx_i = -(x_i - x_j)/|x_i - x_j|^2
    # So grad_i = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    
    # Compute 1/dists^2 with mask
    inv_dists_sq = np.zeros((n, n))
    inv_dists_sq[mask] = 1.0 / (dists[mask] ** 2)
    
    # grad[i] = -sum_j (x_i - x_j) * inv_dists_sq[i,j]
    # = -x_i * sum_j inv_dists_sq[i,j] + sum_j x_j * inv_dists_sq[i,j]
    sum_inv = np.sum(inv_dists_sq, axis=1)
    grad = -pts * sum_inv[:, np.newaxis] + (inv_dists_sq @ pts)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.sqrt(np.sum(pts * pts, axis=1, keepdims=True))
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _momentum_step(pts, grad, lr, momentum, velocity):
    """Momentum gradient descent step on sphere."""
    velocity = momentum * velocity - lr * grad
    new_pts = pts + velocity
    return _project_to_sphere(new_pts), velocity

def construct():
    start_time = time.time()
    
    # Initial points: Fibonacci spiral
    best_pts = _fibonacci_sphere(N)
    best_energy, _ = _log_energy_and_grad(best_pts)
    
    # Try multiple random restarts with momentum + L-BFGS-B hybrid
    np.random.seed(42)
    
    # First, try L-BFGS-B from Fibonacci start
    def energy_only(pts):
        e, _ = _log_energy_and_grad(pts)
        return e
    
    def grad_only(pts):
        _, g = _log_energy_and_grad(pts)
        return g
    
    # Convert to flat representation
    x0 = best_pts.flatten()
    
    try:
        res = minimize(
            energy_only, x0, jac=grad_only, method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if res.success or res.fun < best_energy:
            candidate = _project_to_sphere(res.x.reshape(N, 3))
            e, _ = _log_energy_and_grad(candidate)
            if e < best_energy and np.all(np.isfinite(candidate)):
                best_pts = candidate
                best_energy = e
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT * 0.7:
        return best_pts
    
    # Now try random restarts with momentum GD followed by L-BFGS-B polish
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT * 0.85:
            break
        
        # Random initial points
        pts = np.random.randn(N, 3)
        pts = _project_to_sphere(pts)
        
        # Momentum gradient descent phase
        lr = 0.01
        momentum = 0.9
        velocity = np.zeros_like(pts)
        
        for it in range(300):
            if time.time() - start_time > TIME_LIMIT * 0.85:
                break
            e, g = _log_energy_and_grad(pts)
            pts, velocity = _momentum_step(pts, g, lr, momentum, velocity)
        
        # L-BFGS-B polish
        x0 = pts.flatten()
        try:
            res = minimize(
                energy_only, x0, jac=grad_only, method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_energy:
                candidate = _project_to_sphere(res.x.reshape(N, 3))
                e, _ = _log_energy_and_grad(candidate)
                if e < best_energy and np.all(np.isfinite(candidate)):
                    best_pts = candidate
                    best_energy = e
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() - start_time < TIME_LIMIT * 0.9:
        x0 = best_pts.flatten()
        try:
            res = minimize(
                energy_only, x0, jac=grad_only, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            if res.fun < best_energy:
                candidate = _project_to_sphere(res.x.reshape(N, 3))
                e, _ = _log_energy_and_grad(candidate)
                if e < best_energy and np.all(np.isfinite(candidate)):
                    best_pts = candidate
                    best_energy = e
        except Exception:
            pass
    
    # Ensure points are on unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Check for points too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        # Perturb close points
        min_idx = np.unravel_index(np.argmin(dists[iu]), (N, N))
        perturbation = np.random.randn(3) * 1e-6
        best_pts[min_idx[0]] += perturbation
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts
