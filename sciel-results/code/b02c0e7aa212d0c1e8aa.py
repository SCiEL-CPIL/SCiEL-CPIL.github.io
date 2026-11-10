import numpy as np
import time
import math

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Function to compute energy and gradient
    def energy_and_grad(pts):
        n = len(pts)
        # Pairwise differences
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
        dist = np.linalg.norm(diff, axis=2)  # (n, n)
        
        # Mask diagonal to avoid division by zero
        mask = ~np.eye(n, dtype=bool)
        dist_safe = dist.copy()
        dist_safe[~mask] = 1.0
        
        # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
        # Sum over upper triangle
        triu_indices = np.triu_indices(n, 1)
        energy = -np.sum(np.log(dist_safe[triu_indices]))
        
        # Gradient: grad_i = - sum_{j != i} (x_i - x_j) / d_ij^2
        # Note: The term for pair (i,j) contributes (x_i - x_j)/d^2 to grad_i and (x_j - x_i)/d^2 to grad_j
        # So we can sum over all j != i
        # inv_dist_sq = 1 / dist_safe^2
        inv_dist_sq = np.zeros_like(dist_safe)
        inv_dist_sq[mask] = 1.0 / (dist_safe[mask] ** 2)
        
        # grad[i] = sum_j (x_i - x_j) * inv_dist_sq[i,j]
        # diff[i,j] = x_i - x_j
        # grad = sum_j diff[i,j] * inv_dist_sq[i,j]
        grad = np.einsum('ijk,ik->j', diff, inv_dist_sq)
        
        # Project gradient to tangent plane
        # grad_proj = grad - (grad . x) x
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        grad = grad - dot * pts
        
        return energy, grad

    current_energy, _ = energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    step_size = 0.01
    max_iter = 5000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        energy, grad = energy_and_grad(pts)
        
        # Update points
        new_pts = pts - step_size * grad
        
        # Normalize to unit sphere
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms < 1e-12] = 1.0
        new_pts = new_pts / norms
        
        new_energy, _ = energy_and_grad(new_pts)
        
        if new_energy < energy:
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
            # Increase step size slightly if improving significantly? Or keep.
            # Let's keep step size constant for simplicity, or decay slowly.
            # step_size *= 1.01 
        else:
            step_size *= 0.5
            if step_size < 1e-8:
                break
        
    # Check validity
    if not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_pts
