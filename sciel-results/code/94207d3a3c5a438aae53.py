"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math

def construct():
    # 1. Initialization: Fibonacci Spiral
    # This provides a good spread to avoid local minima in the early stages.
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize strictly
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    
    # 2. Optimization: Vectorized Gradient Descent with Adaptive Step Size
    # Energy E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Gradient for point k: g_k = -sum_{j!=k} (x_k - x_j) / ||x_k - x_j||^2
    # Projected gradient: p_k = g_k - (g_k . x_k) x_k
    
    step_size = 1e-3
    min_step = 1e-8
    max_iter = 2000
    prev_energy = np.inf
    
    for _ in range(max_iter):
        # Compute pairwise differences: (N, N, 3)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        
        # Compute squared distances: (N, N)
        # dist_sq[i, j] = ||x_i - x_j||^2
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero for i=j
        # We only care about i != j
        np.fill_diagonal(dist_sq, 1.0)
        
        # Compute distances
        dist = np.sqrt(dist_sq)
        
        # Compute Energy
        # Sum over upper triangle
        iu = np.triu_indices(N, k=1)
        energy = -np.sum(np.log(dist[iu]))
        
        # Check convergence
        if abs(prev_energy - energy) < 1e-10:
            break
        prev_energy = energy
        
        # Compute Gradient
        # grad[i, j] contribution to x_i from pair (i, j)
        # Term is - (x_i - x_j) / dist_ij^2
        # Note: dist_sq is 1.0 on diagonal, so we can mask later or just let it be 0 if diff is 0
        # diff[i, i] is 0, so term is 0.
        
        # We need sum over j for each i.
        # factor[i, j] = 1 / dist_sq[i, j]
        # But we must exclude diagonal.
        factor = 1.0 / dist_sq
        np.fill_diagonal(factor, 0.0)
        
        # grad[i] = - sum_j factor[i, j] * diff[i, j]
        # diff[i, j] = x_i - x_j
        # So grad = - (factor * diff).sum(axis=1)
        
        # Multiply diff by factor broadcasted
        weighted_diff = diff * factor[:, :, np.newaxis]
        grad = -np.sum(weighted_diff, axis=1)
        
        # Project gradient onto tangent space
        # grad_proj = grad - (grad . x) * x
        grad_dot_x = np.sum(grad * x, axis=1, keepdims=True)
        grad_proj = grad - grad_dot_x * x
        
        # Update step
        # Simple adaptive step: if energy increases, reduce step
        # We don't know new energy yet, so we use a heuristic or backtracking.
        # For simplicity and speed, let's use a fixed decaying step or a very small constant step.
        # Given the landscape is smooth, a small constant step with many iterations works well.
        # Let's try a slightly more sophisticated update:
        # Use the norm of the gradient to scale step?
        # Or just a fixed step that is safe.
        
        # Let's use a simple backtracking line search to ensure decrease.
        # But that requires multiple energy evaluations. 
        # Given N=150, one energy eval is ~150^2 ops. 2000 iters * 150^2 * 10 (line search) might be too slow?
        # 2000 * 22500 * 10 = 450,000,000 ops. NumPy handles this in < 1s.
        
        # Let's implement a simple line search.
        alpha = step_size
        for _ in range(10):
            x_new = x - alpha * grad_proj
            # Normalize
            x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
            
            # Check if any points are too close (invalid)
            # Calculate min distance
            diff_new = x_new[:, np.newaxis, :] - x_new[np.newaxis, :, :]
            dist_sq_new = np.sum(diff_new * diff_new, axis=2)
            np.fill_diagonal(dist_sq_new, 1.0)
            min_dist = np.sqrt(np.min(dist_sq_new))
            
            if min_dist < 1e-9:
                alpha *= 0.5
                continue
                
            # Calculate new energy
            dist_new = np.sqrt(dist_sq_new)
            energy_new = -np.sum(np.log(dist_new[iu]))
            
            if energy_new < energy:
                x = x_new
                break
            else:
                alpha *= 0.5
        
        # Decay step size slightly over time for stability
        step_size *= 0.999
        
        if step_size < min_step:
            break

    # Final normalization
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    
    return x
# EVOLVE-BLOCK-END
