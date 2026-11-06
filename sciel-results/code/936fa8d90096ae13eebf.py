import numpy as np
import time
import math

N = 100

def _log_energy(points):
    """
    Compute the logarithmic energy: sum_{i<j} log(1/d_ij)
    """
    # points shape (N, 3)
    # Compute pairwise distances
    # dist[i, j] = ||points[i] - points[j]||
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # We need sum over i < j of log(1/d_ij)
    # This is - sum over i < j of log(d_ij)
    # Mask upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = dist[mask]
    # Avoid log(0)
    d = np.maximum(d, 1e-10)
    energy = -np.sum(np.log(d))
    return energy

def _gradient(points):
    """
    Compute the gradient of the log-energy E wrt points.
    We want to minimize E, so we move in the direction of -grad(E).
    Alternatively, maximize S = -E. Grad(S) = -Grad(E).
    
    Let's compute grad(S) directly.
    S = sum_{i<j} log(||x_i - x_j||)
    dS/dx_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    
    Then we project this gradient to the tangent space.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Avoid division by zero
    dist_sq = np.maximum(dist_sq, 1e-10)
    
    # grad[i, j] = (x_i - x_j) / dist_sq[i, j]
    # We want sum over j for each i
    # grad_vec[i] = sum_j (x_i - x_j) / dist_sq[i, j]
    # Note: for i=j, term is 0.
    
    # (x_i - x_j) is diff[i, j]
    # We divide diff by dist_sq[..., np.newaxis]
    inv_dist_sq = 1.0 / dist_sq
    # Set diagonal to 0 to avoid self-interaction
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad_vec[i] = sum_j diff[i, j] * inv_dist_sq[i, j]
    grad_vec = np.sum(diff * inv_dist_sq[..., np.newaxis], axis=1)
    
    # Project to tangent space: grad_proj = grad_vec - (grad_vec . x_i) x_i
    # x_i is points[i]
    # dot = sum(grad_vec[i] * points[i])
    dot = np.sum(grad_vec * points, axis=1)
    grad_proj = grad_vec - dot[:, np.newaxis] * points
    
    return grad_proj

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    best_points = None
    best_energy = float('inf')
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    base_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Try a few restarts with different noise levels
    # Or just one long run with adaptive step?
    # Let's do multiple restarts.
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        if restart == 0:
            points = base_points.copy()
        else:
            points = base_points + np.random.normal(0, 0.1, size=(N, 3))
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Gradient ascent on S (minimize E)
        # Step size
        lr = 0.01
        momentum = 0.9
        vel = np.zeros_like(points)
        
        # Run for a fixed number of steps or until time
        steps = 0
        max_steps = 2000
        
        while steps < max_steps and time.time() < deadline:
            grad = _gradient(points)
            
            # Update velocity
            vel = momentum * vel + lr * grad
            points = points + vel
            
            # Project to sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
            steps += 1
            
            # Compute energy periodically
            if steps % 50 == 0:
                energy = _log_energy(points)
                if energy < best_energy:
                    best_energy = energy
                    best_points = points.copy()
        
        # Final check for this restart
        energy = _log_energy(points)
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
            
    if best_points is None:
        return base_points
        
    return best_points
