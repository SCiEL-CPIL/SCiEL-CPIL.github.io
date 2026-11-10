# EVOLVE-BLOCK-START
import numpy as np
import math

N = 150

def _log_energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Avoid division by zero / log(0) for i==j (dist=0)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    # Filter out any distances that are too small
    d = np.maximum(d, 1e-15)
    terms = np.log(1.0 / d)
    return math.fsum(terms)

def _log_energy_grad(points):
    """Compute the gradient of the logarithmic energy with respect to points."""
    n = len(points)
    grad = np.zeros_like(points)
    # Vectorized computation for better performance
    # diff[i, j, k] = points[i, k] - points[j, k]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    dist_safe = np.where(dist > 1e-15, dist, 1e-15)
    
    # Gradient term: factor * diff
    # factor[i, j] = 1 / dist[i, j]^2
    factor = 1.0 / (dist_safe * dist_safe)
    
    # Set diagonal to 0 to avoid self-interaction
    np.fill_diagonal(factor, 0.0)
    
    # grad[i] = sum_j factor[i, j] * (points[i] - points[j])
    # This is equivalent to: sum_j factor[i, j] * diff[i, j]
    # We can compute this using einsum or matrix multiplication
    # grad[i, k] = sum_j factor[i, j] * diff[i, j, k]
    grad = np.einsum('ij,ijk->ik', factor, diff)
    
    return grad

def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _optimize_repulsive(points, max_iters=2000, initial_lr=0.1):
    """
    Optimize points on the sphere using gradient descent with adaptive step size.
    """
    current_points = points.copy()
    current_energy = _log_energy(current_points)
    lr = initial_lr
    min_lr = 1e-6
    
    for _ in range(max_iters):
        grad = _log_energy_grad(current_points)
        
        # Project gradient to tangent space
        # grad_tan = grad - (grad . x) x
        dot = np.sum(grad * current_points, axis=1, keepdims=True)
        grad_tan = grad - dot * current_points
        
        # Check if gradient is small enough
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < 1e-10:
            break
            
        # Update points
        new_points = current_points - lr * grad_tan
        new_points = _project_to_sphere(new_points)
        
        # Check for validity
        if not np.all(np.isfinite(new_points)):
            lr *= 0.5
            continue
            
        # Check minimum distance
        diff = new_points[:, np.newaxis, :] - new_points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        if np.min(dist) < 1e-9:
            lr *= 0.5
            continue
            
        new_energy = _log_energy(new_points)
        
        # Adaptive learning rate
        if new_energy < current_energy:
            current_points = new_points
            current_energy = new_energy
            # Increase lr slightly if we are making good progress
            if lr < 0.5:
                lr *= 1.05
        else:
            # Decrease lr if energy increases
            lr *= 0.5
            if lr < min_lr:
                break
                
    return current_points

def construct():
    # Generate multiple initial configurations
    configs = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_points = _project_to_sphere(fib_points)
    configs.append(fib_points)
    
    # 2. Random points
    rng = np.random.default_rng(42)
    for _ in range(5):
        rand_points = rng.normal(size=(N, 3))
        rand_points = _project_to_sphere(rand_points)
        configs.append(rand_points)
        
    # 3. Perturbed Fibonacci
    for _ in range(5):
        perturbed = fib_points + rng.normal(scale=0.1, size=(N, 3))
        perturbed = _project_to_sphere(perturbed)
        configs.append(perturbed)
        
    # Optimize each configuration and keep the best
    best_points = None
    best_energy = float('inf')
    
    for config in configs:
        optimized = _optimize_repulsive(config, max_iters=500, initial_lr=0.05)
        energy = _log_energy(optimized)
        
        # Check validity
        norms = np.linalg.norm(optimized, axis=1)
        if np.any(norms < 1e-15) or np.any(~np.isfinite(optimized)):
            continue
            
        diff = optimized[:, np.newaxis, :] - optimized[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        if np.min(dist) < 1e-9:
            continue
            
        if energy < best_energy:
            best_energy = energy
            best_points = optimized
            
    # Fallback if all optimizations failed
    if best_points is None:
        best_points = fib_points
        
    return best_points
# EVOLVE-BLOCK-END
