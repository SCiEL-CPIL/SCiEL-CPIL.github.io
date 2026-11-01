import numpy as np
import math
import time

N = 150
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _energy(points):
    """Compute log(1/d) energy = -sum log(d) over all pairs."""
    n = len(points)
    # Compute all pairwise distances
    # d_ij = |x_i - x_j|
    # We want sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Use efficient vectorized computation
    # d2_ij = |x_i - x_j|^2 = 2 - 2*x_i.x_j for unit vectors
    dot = points @ points.T  # n x n
    # d2 = 2 - 2*dot, but diagonal is 0
    d2 = 2.0 - 2.0 * dot
    np.fill_diagonal(d2, 0.0)
    # Clip to avoid log(0)
    d2 = np.maximum(d2, 1e-30)
    # We only want upper triangle (i<j)
    iu = np.triu_indices(n, k=1)
    log_terms = np.log(1.0 / np.sqrt(d2[iu]))
    return math.fsum(log_terms)

def _energy_grad(points):
    """
    Compute energy and gradient.
    Energy E = sum_{i<j} log(1/|x_i - x_j|)
    dE/dx_i = -2 * sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    # Compute pairwise differences and squared distances
    # diff[i,j] = x_i - x_j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # n x n x 3
    d2 = np.sum(diff * diff, axis=2)  # n x n
    np.fill_diagonal(d2, 1.0)  # avoid division by zero on diagonal
    
    # Gradient: grad_i = -2 * sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = -2 * sum_j (x_i - x_j) / d2_ij
    inv_d2 = 1.0 / d2  # n x n
    # grad_i = -2 * sum_j diff[i,j] * inv_d2[i,j]
    grad = -2.0 * np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    # Compute energy
    d2_upper = d2[np.triu_indices(n, k=1)]
    d2_upper = np.maximum(d2_upper, 1e-30)
    log_terms = np.log(1.0 / np.sqrt(d2_upper))
    energy = math.fsum(log_terms)
    
    return energy, grad

def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _fibonacci_init():
    """Fibonacci spiral initial points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Random points on sphere."""
    theta = rng.uniform(0, 2 * np.pi, N)
    z = rng.uniform(-1, 1, N)
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])

def _gradient_descent(points, max_time, rng):
    """
    Gradient descent with adaptive step size on the sphere.
    Returns the best points found.
    """
    points = _project_to_sphere(points.copy())
    energy, grad = _energy_grad(points)
    best_points = points.copy()
    best_energy = energy
    
    # Adaptive step size
    step = 0.01
    min_step = 1e-8
    max_step = 0.5
    patience = 0
    max_patience = 50
    
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        # Compute gradient direction (tangent to sphere)
        # Project gradient to tangent space: grad_tan = grad - (grad . x) * x
        grad_dot_x = np.sum(grad * points, axis=1, keepdims=True)
        grad_tan = grad - grad_dot_x * points
        
        # Take a step
        new_points = points - step * grad_tan
        new_points = _project_to_sphere(new_points)
        
        new_energy, new_grad = _energy_grad(new_points)
        
        if new_energy < energy:
            # Accept step
            points = new_points
            grad = new_grad
            energy = new_energy
            patience = 0
            # Increase step size slightly
            step = min(step * 1.05, max_step)
            
            if energy < best_energy:
                best_energy = energy
                best_points = points.copy()
        else:
            # Reject step, decrease step size
            step *= 0.5
            patience += 1
            if patience > max_patience:
                break
        
        if step < min_step:
            break
    
    return best_points, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with Fibonacci spiral
    points = _fibonacci_init()
    energy, _ = _energy_grad(points)
    best_points = points.copy()
    best_energy = energy
    
    # Do a gradient descent from the Fibonacci start
    remaining = deadline - time.time() - 2.0  # 2s safety margin
    if remaining > 10:
        pts, e = _gradient_descent(points, remaining, None)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Try random restarts
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 5.0:  # 5s safety margin
        # Generate a random perturbation of the best points
        noise_scale = 0.1
        perturbed = best_points + rng.normal(0, noise_scale, (N, 3))
        perturbed = _project_to_sphere(perturbed)
        
        remaining = deadline - time.time() - 3.0
        if remaining < 5:
            break
        
        pts, e = _gradient_descent(perturbed, remaining, rng)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Final check: ensure valid output
    result = _project_to_sphere(best_points)
    # Ensure no points are too close
    for _ in range(10):
        d2 = np.sum((result[:, np.newaxis, :] - result[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(d2, 1.0)
        min_d2 = np.min(d2)
        if min_d2 > (1e-9) ** 2:
            break
        # Nudge the closest pair apart
        iu = np.argmin(d2)
        i, j = divmod(iu, N)
        if i == j:
            break
        # Move them slightly apart
        diff = result[i] - result[j]
        norm_diff = np.linalg.norm(diff)
        if norm_diff < 1e-12:
            diff = rng.normal(0, 1, 3)
        diff = diff / np.linalg.norm(diff)
        result[i] += 0.01 * diff
        result[j] -= 0.01 * diff
        result = _project_to_sphere(result)
    
    return result
