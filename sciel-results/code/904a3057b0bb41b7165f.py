"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute all pairwise distances
    # Using broadcasting for efficiency
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Clip to avoid log of zero or negative
    d = np.maximum(d, 1e-15)
    return math.fsum(math.log(1.0 / di) for di in d)


def _gradient(points):
    """Compute gradient of log energy with respect to points."""
    n = len(points)
    grad = np.zeros_like(points)
    # For each pair (i,j), the contribution to gradient is:
    # d/dx_i sum log(1/|x_i - x_j|) = - (x_i - x_j) / |x_i - x_j|^2
    # d/dx_j sum log(1/|x_i - x_j|) = (x_i - x_j) / |x_i - x_j|^2
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            # Gradient for point i
            grad[i] -= diff / (d * d)
            # Gradient for point j
            grad[j] += diff / (d * d)
    return grad


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize to ensure unit sphere
    points = _project_to_sphere(points)
    
    # Use L-BFGS-B to minimize the log energy
    # We'll use a coordinate-descent style approach with gradient-based optimization
    # Since we need to keep points on the sphere, we use a projected gradient method
    
    best_points = points.copy()
    best_energy = _log_energy(best_points)
    
    # Try a few iterations of gradient descent with projection
    lr = 0.01
    for it in range(500):
        grad = _gradient(best_points)
        # Project gradient to tangent space
        # Tangent projection: g_t = g - (g · x) * x
        grad_t = grad - np.sum(grad * best_points, axis=1, keepdims=True) * best_points
        # Update
        new_points = best_points - lr * grad_t
        new_points = _project_to_sphere(new_points)
        new_energy = _log_energy(new_points)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_points = new_points
            # Decrease learning rate slowly
            lr = max(0.001, lr * 0.999)
        else:
            # Decrease learning rate
            lr = max(0.001, lr * 0.9)
    
    # Additional refinement with smaller steps
    lr = 0.001
    for it in range(200):
        grad = _gradient(best_points)
        grad_t = grad - np.sum(grad * best_points, axis=1, keepdims=True) * best_points
        new_points = best_points - lr * grad_t
        new_points = _project_to_sphere(new_points)
        new_energy = _log_energy(new_points)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_points = new_points
    
    return best_points
# EVOLVE-BLOCK-END
