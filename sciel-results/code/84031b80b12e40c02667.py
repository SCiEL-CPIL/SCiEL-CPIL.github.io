"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _fibonacci_spiral(n, seed_offset=0.0):
    """Generate n points on the unit sphere using a Fibonacci spiral with optional offset."""
    i = np.arange(n) + 0.5 + seed_offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _log_energy_and_grad(X):
    """
    Compute the logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient with respect to X (N x 3).
    
    Energy = -sum_{i<j} log(d_ij)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / d_ij^2
    """
    # Compute all pairwise distances
    # X is (N, 3)
    # diff[i,j] = x_i - x_j
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero and log(0)
    # Set diagonal to 1 to avoid log(1)=0 and division issues, but we mask it out
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Energy: sum_{i<j} -log(d_ij)
    d_pairs = dist[mask]
    energy = -np.sum(np.log(d_pairs))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / d_ij^2
    inv_dist_sq = 1.0 / dist_sq
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(X):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _velocity_verlet_sphere(X, dt=0.01, max_iter=200, tol=1e-10):
    """
    Velocity-Verlet dynamics on the sphere to explore basins.
    Returns (X_final, energy_final).
    """
    X_curr = X.copy()
    V = np.zeros_like(X_curr)  # velocity in R^3
    energy_curr, grad_curr = _log_energy_and_grad(X_curr)
    
    for it in range(max_iter):
        # Project gradient to tangent space
        dot_products = np.sum(grad_curr * X_curr, axis=1)
        grad_tangent = grad_curr - dot_products[:, np.newaxis] * X_curr
        
        # Velocity Verlet: half step
        V = V - 0.5 * dt * grad_tangent
        # Project velocity to tangent space
        V_dot_X = np.sum(V * X_curr, axis=1)
        V = V - V_dot_X[:, np.newaxis] * X_curr
        
        # Full position step
        X_new = X_curr + dt * V
        X_new = _project_to_sphere(X_new)
        
        energy_new, grad_new = _log_energy_and_grad(X_new)
        
        # Half step velocity update
        V = V - 0.5 * dt * grad_new
        # Project velocity again
        V_dot_X = np.sum(V * X_new, axis=1)
        V = V - V_dot_X[:, np.newaxis] * X_new
        
        # Damping to help convergence
        V *= 0.999
        
        X_curr = X_new
        energy_curr = energy_new
        grad_curr = grad_new
        
        # Check convergence
        if np.linalg.norm(grad_tangent) < tol:
            break
    
    return X_curr, energy_curr


def _gradient_descent_sphere(X, max_iter=500, tol=1e-10, deadline=None):
    """
    Riemannian gradient descent with backtracking line search.
    Returns (X_final, energy_final).
    """
    X_curr = X.copy()
    energy_curr, grad_curr = _log_energy_and_grad(X_curr)
    
    step_size = 0.01
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Project gradient to tangent space
        dot_products = np.sum(grad_curr * X_curr, axis=1)
        grad_tangent = grad_curr - dot_products[:, np.newaxis] * X_curr
        
        # Take a step
        X_new = X_curr - step_size * grad_tangent
        X_new = _project_to_sphere(X_new)
        
        energy_new, grad_new = _log_energy_and_grad(X_new)
        
        # Check if energy decreased
        if energy_new < energy_curr:
            X_curr = X_new
            energy_curr = energy_new
            grad_curr = grad_new
            step_size = min(step_size * 1.5, 0.1)
        else:
            step_size *= 0.5
            if step_size < 1e-8:
                break
        
        # Check convergence
        if np.linalg.norm(grad_tangent) < tol:
            break
    
    return X_curr, energy_curr


def _high_precision_polish(X, deadline=None):
    """
    Final high-precision polish pass with very tight tolerance.
    """
    X_curr = X.copy()
    energy_curr, grad_curr = _log_energy_and_grad(X_curr)
    
    step_size = 0.001
    max_iter = 2000
    tol = 1e-14
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        dot_products = np.sum(grad_curr * X_curr, axis=1)
        grad_tangent = grad_curr - dot_products[:, np.newaxis] * X_curr
        
        X_new = X_curr - step_size * grad_tangent
        X_new = _project_to_sphere(X_new)
        
        energy_new, grad_new = _log_energy_and_grad(X_new)
        
        if energy_new < energy_curr:
            X_curr = X_new
            energy_curr = energy_new
            grad_curr = grad_new
        else:
            step_size *= 0.5
            if step_size < 1e-15:
                break
        
        if np.linalg.norm(grad_tangent) < tol:
            break
    
    return X_curr, energy_curr


def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin
    
    best_X = _fibonacci_spiral(N)
    best_energy, _ = _log_energy_and_grad(best_X)
    
    # Phase 1: Multi-start with gradient descent
    n_starts = 25
    
    for start_idx in range(n_starts):
        if time.time() > deadline - 30.0:
            break
        
        # Generate initial configuration
        if start_idx == 0:
            X = _fibonacci_spiral(N)
        elif start_idx < 10:
            # Perturbed Fibonacci spiral
            offset = np.random.uniform(-0.5, 0.5)
            X = _fibonacci_spiral(N, seed_offset=offset)
            X += 0.01 * np.random.randn(N, 3)
            X = _project_to_sphere(X)
        else:
            # Random uniform on sphere
            X = np.random.randn(N, 3)
            X = _project_to_sphere(X)
        
        # Gradient descent
        X_curr, energy_curr = _gradient_descent_sphere(X, max_iter=500, tol=1e-10, deadline=deadline - 30.0)
        
        # Velocity-Verlet dynamics for basin exploration
        if time.time() < deadline - 25.0:
            X_vv, energy_vv = _velocity_verlet_sphere(X_curr, dt=0.005, max_iter=100, tol=1e-8)
            if energy_vv < energy_curr:
                X_curr, energy_curr = X_vv, energy_vv
                # Polish after VV
                X_curr, energy_curr = _gradient_descent_sphere(X_curr, max_iter=200, tol=1e-12, deadline=deadline - 25.0)
        
        # Update best
        if energy_curr < best_energy:
            best_energy = energy_curr
            best_X = X_curr.copy()
    
    # Phase 2: Final high-precision polish
    if time.time() < deadline - 5.0:
        X_polished, energy_polished = _high_precision_polish(best_X, deadline=deadline - 2.0)
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_X = X_polished
    
    # Ensure all points are on the unit sphere and valid
    best_X = _project_to_sphere(best_X)
    
    # Check for any issues
    norms = np.linalg.norm(best_X, axis=1)
    if not np.all(np.isfinite(best_X)) or np.any(norms < 1e-9):
        # Fallback to Fibonacci spiral
        best_X = _fibonacci_spiral(N)
    
    return best_X
# EVOLVE-BLOCK-END
