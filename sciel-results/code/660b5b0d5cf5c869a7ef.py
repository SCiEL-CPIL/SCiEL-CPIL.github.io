"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    # points: (N, 3) array on unit sphere
    n = points.shape[0]
    # Compute all pairwise distances
    # Using broadcasting for clarity, but for N=100 this is fine
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # Energy is sum over i<j of log(1/d) = - sum over i<j log(d)
    # We only need upper triangle
    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    # Clip to avoid log(0) if any points are too close
    d_upper = np.clip(d_upper, 1e-15, None)
    return -math.fsum(np.log(d_upper).tolist())


def _gradient(points):
    """Compute gradient of the logarithmic energy w.r.t. points on the sphere.
    
    The gradient in R^3 is sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    Then we project to the tangent space: grad_tangent = grad - (grad . x_i) x_i
    """
    n = points.shape[0]
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = points[i] - points[j]
            dist_sq = np.dot(d, d)
            if dist_sq < 1e-20:
                continue
            grad[i] += d / dist_sq
    # Project to tangent space
    for i in range(n):
        proj = np.dot(grad[i], points[i])
        grad[i] -= proj * points[i]
    return grad


def _retract(points, grad, step):
    """Retract gradient onto the sphere using exponential map approximation.
    
    For small steps, we can use: x_new = normalize(x - step * grad)
    This is a simple retraction.
    """
    new_points = points - step * grad
    # Normalize to unit sphere
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-15, None)
    new_points /= norms
    return new_points


def _fibonacci_sphere(n):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    deadline = time.time() + 140  # 140s safety margin before 150s kill
    
    # Start with Fibonacci spiral
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_energy = _energy(points)
    
    # Gradient descent with adaptive step size
    step = 0.01
    min_step = 1e-8
    max_iter = 50000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        grad = _gradient(points)
        
        # Check if gradient is small enough
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Take a step
        new_points = _retract(points, grad, step)
        new_energy = _energy(new_points)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_points = new_points.copy()
            # Increase step slightly if we made good progress
            step = min(step * 1.05, 0.1)
        else:
            # Decrease step
            step *= 0.5
            if step < min_step:
                break
        
        points = new_points
    
    # Final polish with smaller steps
    if time.time() < deadline - 5:
        step = 1e-5
        for it in range(10000):
            if time.time() > deadline:
                break
            grad = _gradient(best_points)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            new_points = _retract(best_points, grad, step)
            new_energy = _energy(new_points)
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = new_points.copy()
            else:
                step *= 0.5
                if step < 1e-12:
                    break
    
    return best_points
# EVOLVE-BLOCK-END
