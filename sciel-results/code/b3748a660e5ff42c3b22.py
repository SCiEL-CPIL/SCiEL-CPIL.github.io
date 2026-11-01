"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math


def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    pts: (N, 3) array of points on the unit sphere.
    Returns (energy, grad) where grad is the Euclidean gradient of the energy
    with respect to the point coordinates (not projected to the tangent space).
    """
    n = pts.shape[0]
    # Pairwise distances
    # d[i,j] = ||pts[i] - pts[j]||
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    d = np.linalg.norm(diff, axis=2)  # (N, N)
    np.fill_diagonal(d, 1.0)  # avoid log(0)

    # Energy: sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    # Use symmetric matrix, take upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    log_d = np.log(d)
    energy = -np.sum(log_d[mask])

    # Gradient: d/dx_i log(1/d_ij) = d/dx_i (-log(d_ij))
    # d_ij = ||x_i - x_j||
    # d/dx_i log(d_ij) = (x_i - x_j) / d_ij^2
    # So d/dx_i (-log(d_ij)) = -(x_i - x_j) / d_ij^2 = (x_j - x_i) / d_ij^2
    #
    # grad_i = sum_{j != i} (x_j - x_i) / d_ij^2
    #
    # Let's compute this efficiently.
    # grad[i] = sum_j (pts[j] - pts[i]) / d[i,j]^2
    #
    # We can compute a matrix W where W[i,j] = 1/d[i,j]^2 for i!=j, 0 for i==j
    inv_d2 = np.zeros_like(d)
    np.fill_diagonal(inv_d2, 0.0)
    inv_d2[mask] = 1.0 / (d[mask] ** 2)
    # Also fill lower triangle
    inv_d2[np.triu(np.ones((n, n), dtype=bool), k=1).T] = 1.0 / (d[np.triu(np.ones((n, n), dtype=bool), k=1).T] ** 2)

    # grad[i] = sum_j inv_d2[i,j] * (pts[j] - pts[i])
    # = sum_j inv_d2[i,j] * pts[j] - pts[i] * sum_j inv_d2[i,j]
    row_sum = np.sum(inv_d2, axis=1)  # (N,)
    grad = inv_d2 @ pts - pts * row_sum[:, np.newaxis]

    return energy, grad


def _project_tangent(pts, grad):
    """
    Project the Euclidean gradient onto the tangent space of the sphere at each point.
    For a point x on the unit sphere, the tangent space is {v : x . v = 0}.
    The projection of a vector g onto the tangent space is g - (g . x) x.
    """
    # grad: (N, 3), pts: (N, 3)
    dot = np.sum(grad * pts, axis=1)  # (N,)
    return grad - dot[:, np.newaxis] * pts


def _exponential_map(pts, tangent_vecs, step_size):
    """
    Move points on the sphere using the exponential map.
    For a point x and tangent vector v with step size h, the new point is:
    x_new = cos(h*||v||) * x + sin(h*||v||) / ||v|| * v
    For small h*||v||, this is approximately x + h*v (with normalization).
    """
    v_norms = np.linalg.norm(tangent_vecs, axis=1)  # (N,)
    # Avoid division by zero
    v_norms_safe = np.where(v_norms > 1e-15, v_norms, 1.0)
    v_dirs = tangent_vecs / v_norms_safe[:, np.newaxis]

    angles = step_size * v_norms  # (N,)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles) / v_norms_safe

    new_pts = cos_a[:, np.newaxis] * pts + sin_a[:, np.newaxis] * tangent_vecs

    # Normalize to ensure they're on the unit sphere
    new_norms = np.linalg.norm(new_pts, axis=1)
    new_norms_safe = np.where(new_norms > 1e-15, new_norms, 1.0)
    new_pts = new_pts / new_norms_safe[:, np.newaxis]

    return new_pts


def _fibonacci_sphere(n):
    """Generate initial points using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Initial points: Fibonacci spiral
    pts = _fibonacci_sphere(N)

    # Compute initial energy
    energy, grad = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = energy

    # Riemannian gradient descent with adaptive step size
    step_size = 0.01
    momentum = np.zeros_like(pts)
    mu = 0.9  # momentum coefficient
    patience = 0
    max_patience = 50
    min_step = 1e-8
    max_step = 0.1

    while time.time() < deadline:
        # Project gradient to tangent space
        tan_grad = _project_tangent(pts, grad)

        # Update momentum
        momentum = mu * momentum + (1 - mu) * tan_grad

        # Try the step
        new_pts = _exponential_map(pts, momentum, step_size)

        new_energy, new_grad = _energy_and_grad(new_pts)

        if new_energy < energy:
            # Accept step
            pts = new_pts
            energy = new_energy
            grad = new_grad
            patience = 0

            # Increase step size if we've been improving
            if patience < 5:
                step_size = min(step_size * 1.05, max_step)
        else:
            # Reject step, decrease step size
            step_size *= 0.5
            patience += 1

            if step_size < min_step or patience > max_patience:
                # Try a different approach: reduce momentum and retry
                mu = max(mu * 0.9, 0.5)
                momentum *= 0.5
                step_size = min_step * 10
                patience = 0

                if mu < 0.51 and step_size < min_step * 2:
                    # Stuck, break
                    break

        # Update best
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()

        # Check for convergence
        if np.linalg.norm(tan_grad) < 1e-10:
            break

    # Final polish: run a few more iterations with very small steps
    step_size = 1e-6
    for _ in range(100):
        if time.time() > deadline:
            break
        energy, grad = _energy_and_grad(pts)
        tan_grad = _project_tangent(pts, grad)
        new_pts = _exponential_map(pts, tan_grad, step_size)
        new_energy, _ = _energy_and_grad(new_pts)
        if new_energy < energy:
            pts = new_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()

    return best_pts
# EVOLVE-BLOCK-END
