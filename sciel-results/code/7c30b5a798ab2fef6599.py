"""Optimized logarithmic energy minimization for 150 points on the sphere."""
import numpy as np
import time
from scipy.optimize import minimize, Bounds

N = 150
TIME_LIMIT = 140.0  # safety margin before 150s kill


def _energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient for points on the sphere.
    Energy = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    """
    # pts: (N, 3) array of unit vectors
    n = len(pts)
    
    # Compute pairwise distances
    # dist[i,j] = |pts[i] - pts[j]|
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid log(0) for i==j
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of log(1/d) = -sum over i<j of log(d)
    # Use upper triangle only
    iu = np.triu_indices(n, k=1)
    d_ij = dist[iu]
    
    energy = -np.sum(np.log(d_ij))
    
    # Gradient: d/dx_i of [-log(|x_i - x_j|)] for each j != i
    # d/dx_i log(|x_i - x_j|) = (x_i - x_j) / |x_i - x_j|^2
    # So d/dx_i [-log(|x_i - x_j|)] = -(x_i - x_j) / |x_i - x_j|^2
    # For the full sum, grad_i = sum_{j != i} -(x_i - x_j) / |x_i - x_j|^2
    #                         = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    
    # Compute gradient efficiently
    # diff[i,j] = x_i - x_j, dist[i,j] = |x_i - x_j|
    # grad_i = sum_j (x_j - x_i) / dist[i,j]^2 = -sum_j diff[i,j] / dist[i,j]^2
    
    # dist^2
    dist2 = dist ** 2
    np.fill_diagonal(dist2, 1.0)  # avoid division by zero
    
    # grad[i] = -sum_j diff[i,j] / dist2[i,j]
    # diff[i,j] / dist2[i,j] is a (N, N, 3) array
    # We want sum over j for each i
    inv_dist2 = 1.0 / dist2
    inv_dist2[np.eye(n, dtype=bool)] = 0.0
    
    # grad[i] = -sum_j (x_i - x_j) * inv_dist2[i,j]
    # = -x_i * sum_j inv_dist2[i,j] + sum_j x_j * inv_dist2[i,j]
    
    # More directly:
    # grad = -np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    # But this includes the diagonal which we zeroed, so it's fine
    
    grad = -np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    # Project gradient onto tangent space: grad_tangent = grad - (grad · x_i) * x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - dot * pts
    
    return energy, grad_tangent


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral(n, seed_offset=0.0):
    """Generate Fibonacci spiral points with optional offset for diversity."""
    i = np.arange(n) + 0.5 + seed_offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_initialization(n, rng):
    """Random points on the sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def construct():
    start_time = time.time()
    
    # Initial guess: Fibonacci spiral
    best_pts = _fibonacci_spiral(N)
    best_energy, _ = _energy_and_grad(best_pts)
    
    # Try multiple restarts with different initializations
    rng = np.random.default_rng(42)
    
    # Define the flat-space objective for L-BFGS-B
    def flat_objective(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, g = _energy_and_grad(pts)
        return e, g
    
    # Use L-BFGS-B with the sphere constraint handled via projection
    # Actually, let's use a parameterization that stays on the sphere
    # Better: use the gradient projected onto the tangent space directly
    
    # We'll do gradient descent with backtracking line search in the tangent space
    def tangent_step(pts, grad, lr):
        """Take a step in the tangent direction and reproject."""
        new_pts = pts - lr * grad
        return _project_to_sphere(new_pts)
    
    def optimize_from(pts, max_iter=2000, tol=1e-10, time_limit=100.0):
        """Gradient descent with adaptive learning rate on the sphere."""
        pts = pts.copy()
        energy, grad = _energy_and_grad(pts)
        best_local_energy = energy
        best_local_pts = pts.copy()
        
        lr = 0.01
        prev_grad_norm = np.linalg.norm(grad)
        
        for it in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < tol:
                break
            
            # Adaptive learning rate
            if it > 0 and prev_grad_norm > 0:
                # Simple adaptation: if energy increased, reduce lr
                pass
            
            new_pts = tangent_step(pts, grad, lr)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < energy:
                pts = new_pts
                energy = new_energy
                grad = new_grad
                if energy < best_local_energy:
                    best_local_energy = energy
                    best_local_pts = pts.copy()
                # Increase lr slightly if we made good progress
                lr = min(lr * 1.1, 0.1)
            else:
                # Reduce learning rate
                lr *= 0.5
                if lr < 1e-8:
                    break
            
            prev_grad_norm = np.linalg.norm(grad)
        
        return best_local_pts, best_local_energy
    
    # First, optimize from the Fibonacci spiral
    if time.time() - start_time < 50:
        opt_pts, opt_energy = optimize_from(best_pts, max_iter=3000, time_limit=45.0)
        if opt_energy < best_energy:
            best_pts = opt_pts
            best_energy = opt_energy
    
    # Try several random restarts
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() - start_time > 130:
            break
        
        # Different random initializations
        if r == 0:
            init_pts = _random_initialization(N, rng)
        elif r == 1:
            init_pts = _fibonacci_spiral(N, seed_offset=0.3)
        elif r == 2:
            init_pts = _fibonacci_spiral(N, seed_offset=0.7)
        elif r == 3:
            # Perturbed Fibonacci
            init_pts = _fibonacci_spiral(N)
            noise = rng.standard_normal((N, 3)) * 0.1
            init_pts = _project_to_sphere(init_pts + noise)
        elif r == 4:
            # Another perturbation
            init_pts = _fibonacci_spiral(N)
            noise = rng.standard_normal((N, 3)) * 0.2
            init_pts = _project_to_sphere(init_pts + noise)
        else:
            init_pts = _random_initialization(N, rng)
        
        # Quick gradient descent
        remaining = 130 - (time.time() - start_time)
        if remaining < 5:
            break
        
        time_per_restart = min(remaining / (n_restarts - r), 12.0)
        opt_pts, opt_energy = optimize_from(init_pts, max_iter=1500, time_limit=time_per_restart)
        
        if opt_energy < best_energy:
            best_pts = opt_pts
            best_energy = opt_energy
    
    # Final polishing with more iterations if time permits
    remaining = 135 - (time.time() - start_time)
    if remaining > 5:
        opt_pts, opt_energy = optimize_from(best_pts, max_iter=5000, time_limit=remaining)
        if opt_energy < best_energy:
            best_pts = opt_pts
            best_energy = opt_energy
    
    # Ensure validity: no points closer than 1e-9, all finite
    # Check and fix if needed
    dist = np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dist, 1.0)
    
    if np.any(~np.isfinite(best_pts)):
        # Fallback to Fibonacci spiral
        best_pts = _fibonacci_spiral(N)
    
    # Check for too-close points and separate them slightly
    min_dist = np.min(dist)
    if min_dist < 1e-9:
        # Find the closest pair and separate
        iu = np.triu_indices(N, k=1)
        d_flat = dist[iu]
        min_idx = np.argmin(d_flat)
        i, j = iu[0][min_idx], iu[1][min_idx]
        # Push them apart slightly
        diff = best_pts[i] - best_pts[j]
        norm_diff = np.linalg.norm(diff)
        if norm_diff < 1e-15:
            # Orthogonal direction
            ortho = np.array([1.0, 0.0, 0.0])
            if abs(best_pts[i][0]) > 0.9:
                ortho = np.array([0.0, 1.0, 0.0])
            ortho = ortho - np.dot(ortho, best_pts[i]) * best_pts[i]
            ortho = ortho / np.linalg.norm(ortho)
            diff = ortho
        else:
            diff = diff / norm_diff
        best_pts[i] = _project_to_sphere(best_pts[i] + 1e-8 * diff)
        best_pts[j] = _project_to_sphere(best_pts[j] - 1e-8 * diff)
    
    return best_pts
