"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on sphere."""
    n = len(pts)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Avoid division by zero and log(0)
    dist = np.maximum(dist, 1e-15)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # Use upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    log_dist = np.log(dist[iu])
    energy = -np.sum(log_dist)
    
    # Gradient: d/dx_i [-log(|x_i - x_j|)] = -(x_i - x_j) / |x_i - x_j|^2
    # Sum over all j != i
    # grad_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    inv_dist_sq = 1.0 / (dist * dist)
    # Set diagonal to 0
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -sum_j (pts[i] - pts[j]) * inv_dist_sq[i,j]
    # = -sum_j pts[i]*inv_dist_sq[i,j] + sum_j pts[j]*inv_dist_sq[i,j]
    # = -pts[i] * sum_j inv_dist_sq[i,j] + sum_j pts[j] * inv_dist_sq[i,j]
    
    row_sums = np.sum(inv_dist_sq, axis=1)
    grad = -pts * row_sums[:, np.newaxis] + inv_dist_sq @ pts
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.sqrt(np.sum(pts * pts, axis=1))
    norms = np.maximum(norms, 1e-15)
    return pts / norms[:, np.newaxis]


def _fibonacci_spiral(n, rng=None):
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere_points(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.sqrt(np.sum(pts * pts, axis=1))
    return pts / norms[:, np.newaxis]


def _optimize_from_start(pts0, max_iter=2000, time_limit=None, start_time=None):
    """Gradient descent with line search on sphere."""
    if start_time is None:
        start_time = time.time()
    
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    # Adaptive step size
    step = 0.1
    min_step = 1e-10
    max_step = 1.0
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        energy, grad = _energy_and_grad(pts)
        
        # Project gradient to tangent space: g_tan = g - (g . x) * x
        dot = np.sum(grad * pts, axis=1)
        grad_tan = grad - dot[:, np.newaxis] * pts
        
        # If gradient is very small, we're likely at a local minimum
        grad_norm = np.sqrt(np.sum(grad_tan * grad_tan))
        if grad_norm < 1e-12:
            break
        
        # Try to take a step
        accepted = False
        for trial in range(20):
            new_pts = pts + step * grad_tan
            new_pts = _project_to_sphere(new_pts)
            new_energy = _energy_and_grad(new_pts)[0]
            
            # Check for improvement
            if new_energy < energy - 1e-12:
                pts = new_pts
                accepted = True
                break
            step *= 0.5
        
        if not accepted:
            step *= 0.5
            if step < min_step:
                break
        else:
            # Occasionally increase step
            if it % 50 == 0 and step < max_step:
                step = min(step * 1.2, max_step)
        
        # Track best
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral
    pts0 = _fibonacci_spiral(N)
    
    best_pts = pts0.copy()
    best_energy = _energy_and_grad(pts0)[0]
    
    # Optimize from the initial configuration
    pts_opt, energy_opt = _optimize_from_start(pts0, max_iter=5000, 
                                                 time_limit=100.0, 
                                                 start_time=start_time)
    if energy_opt < best_energy:
        best_energy = energy_opt
        best_pts = pts_opt
    
    # Try several random restarts if time permits
    num_restarts = 10
    for r in range(num_restarts):
        if (time.time() - start_time) > deadline - 20.0:
            break
        
        # Create perturbed starting point
        if r == 0:
            # Slight perturbation of best found
            pts_start = best_pts + 0.1 * rng.standard_normal((N, 3))
        else:
            # Mix of Fibonacci and random
            alpha = rng.uniform(0.3, 0.7)
            pts_fib = _fibonacci_spiral(N)
            pts_rand = _random_sphere_points(N, rng)
            pts_start = alpha * pts_fib + (1 - alpha) * pts_rand
        
        pts_start = _project_to_sphere(pts_start)
        
        remaining = deadline - 10.0 - (time.time() - start_time)
        if remaining < 5.0:
            break
        
        pts_opt, energy_opt = _optimize_from_start(pts_start, max_iter=3000,
                                                     time_limit=remaining,
                                                     start_time=start_time)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    return _project_to_sphere(best_pts)
# EVOLVE-BLOCK-END
