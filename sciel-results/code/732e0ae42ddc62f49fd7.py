"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    dots = np.dot(pts, pts.T)
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    dist = np.sqrt(dist_sq)
    log_inv_dist = -np.log(dist)
    np.fill_diagonal(log_inv_dist, 0.0)
    triu_indices = np.triu_indices(n, k=1)
    energy = math.fsum(log_inv_dist[triu_indices].tolist())
    return energy


def _project_to_sphere(pts):
    """Normalize each point to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _energy_and_grad(pts):
    """Vectorized energy and gradient for points on the unit sphere."""
    # pts: (N, 3)
    dots = np.dot(pts, pts.T)
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    dist = np.sqrt(dist_sq)
    log_inv_dist = -np.log(dist)
    np.fill_diagonal(log_inv_dist, 0.0)
    triu_indices = np.triu_indices(N, k=1)
    energy = np.sum(log_inv_dist[triu_indices])
    
    # Gradient: dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} (x_i - x_j) / dist_sq[i,j]
    # Build matrix of (x_i - x_j) / dist_sq[i,j]
    # diff[i,j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    # Divide by dist_sq
    dist_sq_safe = np.maximum(dist_sq, 1e-15)
    np.fill_diagonal(dist_sq_safe, 1.0)
    # grad[i] = sum_j (x_i - x_j) / dist_sq[i,j]
    # We want to sum over j for each i
    # So we divide each diff[i,j,:] by dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq_safe
    # Zero out diagonal
    np.fill_diagonal(inv_dist_sq, 0.0)
    # Multiply diff by inv_dist_sq[..., np.newaxis]
    weighted_diff = diff * inv_dist_sq[..., np.newaxis]
    # Sum over j (axis=1)
    grad = np.sum(weighted_diff, axis=1)  # (N, 3)
    
    # Project gradient to tangent space
    g_dot_x = np.sum(grad * pts, axis=1, keepdims=True)  # (N, 1)
    grad = grad - g_dot_x * pts
    
    return energy, grad


def _fibonacci_spiral(n):
    """Generate points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _gradient_descent(pts, max_iter=20000, initial_lr=0.05):
    """Gradient descent with projection to sphere and adaptive learning rate."""
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    lr = initial_lr
    prev_energy = best_energy
    patience = 0
    
    for step in range(max_iter):
        energy, grad = _energy_and_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
        
        # Adaptive learning rate: reduce if no improvement
        if patience > 100:
            lr *= 0.5
            patience = 0
            if lr < 1e-8:
                break
        
        # Update
        pts = pts - lr * grad
        
        # Project back to sphere
        pts = _project_to_sphere(pts)
        
        # Periodic check with more accurate energy
        if step % 500 == 0:
            acc_energy = _log_energy(pts)
            if acc_energy < best_energy:
                best_energy = acc_energy
                best_pts = pts.copy()
    
    return best_pts, best_energy


def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    
    # Phase 1: Gradient descent
    pts, best_energy = _gradient_descent(pts, max_iter=15000, initial_lr=0.05)
    
    # Phase 2: Use L-BFGS-B with a reparameterization
    # We'll optimize in R^3 with a penalty for not being on the sphere,
    # or better: use the fact that we can optimize on the sphere via
    # exponential map or just use the flat coordinates with normalization
    # in the objective.
    
    def obj(x_flat):
        p = x_flat.reshape(N, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
    
    def obj_grad(x_flat):
        p = x_flat.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_grad(p)
        # The gradient with respect to x_flat needs to account for the projection
        # For simplicity, we'll use the projected gradient and hope L-BFGS-B
        # can handle the non-smoothness from normalization, or we'll just
        # use the gradient as is since the points are already normalized.
        # Actually, this is not quite right. Let's use a different approach:
        # optimize using the fact that we can parameterize the sphere.
        # For now, return the tangent-space gradient as an approximation.
        return g.flatten()
    
    try:
        res = minimize(
            obj,
            pts.flatten(),
            jac=obj_grad,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        if res.success or res.fun < best_energy:
            pts = _project_to_sphere(res.x.reshape(N, 3))
            e = _log_energy(pts)
            if e < best_energy:
                best_energy = e
    except Exception:
        pass
    
    # Phase 3: Final gradient descent polish
    pts, final_energy = _gradient_descent(pts, max_iter=5000, initial_lr=0.01)
    if final_energy < best_energy:
        best_energy = final_energy
    
    return _project_to_sphere(pts)

# EVOLVE-BLOCK-END
