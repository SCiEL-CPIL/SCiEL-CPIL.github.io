"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize


def _fibonacci_sphere():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # normalize to be safe
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _energy_and_grad(x):
    """
    x: (N, 3) on unit sphere (approximately).
    Returns energy (sum_{i<j} log(1/||x_i - x_j||)) and gradient w.r.t. x (3D),
    then we project gradient to tangent space.
    """
    N = x.shape[0]
    # Compute pairwise distances
    # diff[i,j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist2, 1.0)  # avoid log(0)
    dist = np.sqrt(dist2)  # (N, N)

    # Energy: sum over i<j of log(1/dist_ij) = - sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)

    # Gradient: d/dx_i [ - log(dist_ij) ] for each pair
    # d/dx_i log(||x_i - x_j||) = (x_i - x_j) / ||x_i - x_j||^2
    # So d/dx_i [-log(dist_ij)] = -(x_i - x_j) / dist_ij^2 = (x_j - x_i) / dist_ij^2
    #
    # We need grad_energy w.r.t. x_i = sum_{j != i} (x_j - x_i) / dist_ij^2
    #
    # Let's compute this efficiently.
    # For each pair (i,j) with i<j, the contribution to grad[i] is (x_j - x_i)/dist_ij^2
    # and to grad[j] is (x_i - x_j)/dist_ij^2
    grad = np.zeros_like(x)
    # Use vectorized approach
    # dist2 is (N,N), we need to avoid division by zero on diagonal
    inv_dist2 = np.zeros_like(dist2)
    mask = ~np.eye(N, dtype=bool)
    inv_dist2[mask] = 1.0 / dist2[mask]
    
    # grad[i] = sum_j (x_j - x_i) * inv_dist2[i,j]
    # = sum_j x_j * inv_dist2[i,j] - x_i * sum_j inv_dist2[i,j]
    sum_inv = np.sum(inv_dist2, axis=1, keepdims=True)  # (N, 1)
    # sum_j x_j * inv_dist2[i,j] = inv_dist2 @ x  (but inv_dist2 is symmetric)
    sum_x_inv = inv_dist2 @ x  # (N, 3)
    grad = sum_x_inv - x * sum_inv

    return energy, grad


def _project_to_sphere(x):
    """Re-normalize points to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _tangent_grad(x, grad):
    """
    Project gradient to tangent space at each point.
    tangent_grad_i = grad_i - (grad_i . x_i) * x_i
    """
    dot = np.sum(grad * x, axis=1, keepdims=True)
    return grad - dot * x


def _line_search(x, grad_tangent, energy_current, max_steps=50):
    """
    Simple backtracking line search on the sphere.
    We move along the geodesic: x_new = (x + t * grad_tangent) / ||x + t * grad_tangent||
    """
    t = 0.1
    for _ in range(max_steps):
        x_trial = x + t * grad_tangent
        x_trial = _project_to_sphere(x_trial)
        e_trial, _ = _energy_and_grad(x_trial)
        if e_trial < energy_current - 1e-10 * abs(energy_current) * t:
            return x_trial, e_trial, t
        t *= 0.5
    return x, energy_current, t


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # leave 10s safety margin

    # Start from Fibonacci sphere
    x = _fibonacci_sphere()
    energy, _ = _energy_and_grad(x)
    best_x = x.copy()
    best_energy = energy

    # Local optimization loop
    max_iter = 500
    for it in range(max_iter):
        if time.time() > deadline:
            break

        energy, grad = _energy_and_grad(x)
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()

        # Check convergence
        grad_tangent = _tangent_grad(x, grad)
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-12:
            break

        # Normalize gradient for step size
        grad_tangent_normed = grad_tangent / grad_norm

        x_new, e_new, t_used = _line_search(x, grad_tangent_normed, energy)
        if np.allclose(x_new, x, atol=1e-15):
            break
        x = x_new

    # Final polish with tighter tolerance
    if time.time() < deadline - 20.0:
        for it in range(200):
            if time.time() > deadline:
                break
            energy, grad = _energy_and_grad(x)
            if energy < best_energy:
                best_energy = energy
                best_x = x.copy()
            grad_tangent = _tangent_grad(x, grad)
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-14:
                break
            grad_tangent_normed = grad_tangent / grad_norm
            x_new, e_new, t_used = _line_search(x, grad_tangent_normed, energy)
            if np.allclose(x_new, x, atol=1e-16):
                break
            x = x_new

    # Ensure final result is on the sphere
    best_x = _project_to_sphere(best_x)
    return best_x
# EVOLVE-BLOCK-END
