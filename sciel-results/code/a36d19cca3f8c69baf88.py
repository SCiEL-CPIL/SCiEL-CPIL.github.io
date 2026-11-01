import numpy as np
import time
import math

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Define objective and gradient with penalty for sphere constraint
    # We optimize in R^3 with a penalty term to keep points on the sphere.
    # E = sum_{i<j} log(1/||xi-xj||) + lambda * sum_i (||xi||^2 - 1)^2
    
    lambda_penalty = 1000.0
    
    def energy_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        
        # Compute pairwise distances
        # dist[i, j] = ||x_i - x_j||
        # Using broadcasting
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero (should not happen with good init, but safe)
        dist_sq[dist_sq < 1e-12] = 1e-12
        dist = np.sqrt(dist_sq)
        
        # Energy: sum_{i<j} log(1/dist) = - sum_{i<j} log(dist)
        # We sum over i<j.
        # Create a mask for upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # Log energy
        log_dists = np.log(dist)
        energy = -np.sum(log_dists[mask])
        
        # Penalty energy
        norms_sq = np.sum(x * x, axis=1)
        penalty = lambda_penalty * np.sum((norms_sq - 1.0) ** 2)
        
        total_energy = energy + penalty
        
        # Gradient
        # Grad of energy part:
        # d/dx_i sum_{j != i} -log(||x_i - x_j||)
        # = sum_{j != i} - (1/||x_i - x_j||) * (x_i - x_j) / ||x_i - x_j||
        # = sum_{j != i} -(x_i - x_j) / ||x_i - x_j||^2
        
        # We can compute this using the diff matrix.
        # grad_energy[i, :] = - sum_{j != i} (x_i - x_j) / dist_sq[i, j]
        
        # diff[i, j] = x_i - x_j
        # We want sum over j != i of -diff[i, j] / dist_sq[i, j]
        
        # Create a matrix of 1/dist_sq, set diagonal to 0
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=(dist_sq > 1e-12))
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # grad_energy[i] = - sum_j diff[i, j] * inv_dist_sq[i, j]
        # diff is (N, N, 3), inv_dist_sq is (N, N)
        # We can do: -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        grad_energy = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Grad of penalty part:
        # d/dx_i lambda * (||x_i||^2 - 1)^2 = lambda * 2(||x_i||^2 - 1) * 2 x_i = 4 lambda x_i (||x_i||^2 - 1)
        grad_penalty = 4.0 * lambda_penalty * x * (norms_sq - 1.0)[:, np.newaxis]
        
        total_grad = grad_energy + grad_penalty
        
        return total_energy, total_grad.flatten()

    # 3. Optimization
    # Use L-BFGS-B
    # We need to import scipy
    from scipy.optimize import minimize
    
    x_flat = x.flatten()
    
    # We might run out of time, so we need to be careful.
    # L-BFGS-B is fast. Let's try to run it.
    
    try:
        res = minimize(
            energy_and_grad, 
            x_flat, 
            method='L-BFGS-B', 
            jac=True, # Function returns (f, g)
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        # Check if it finished within time
        if time.time() > deadline:
             # Return best found so far? 
             # If it didn't finish, res might not be optimal. 
             # But L-BFGS usually converges fast for N=150.
             pass
             
        x_opt = res.x.reshape(N, 3)
        
    except Exception as e:
        # Fallback to initial if optimizer fails
        x_opt = x
        print(f"Optimizer failed: {e}")
        
    # 4. Normalize points to unit sphere
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-12] = 1.0
    x_final = x_opt / norms
    
    return x_final
