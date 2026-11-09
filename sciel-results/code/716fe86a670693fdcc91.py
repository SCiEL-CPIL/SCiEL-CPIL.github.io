"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def construct():
    # 1. Initial Configuration: Fibonacci Spiral
    # This is a good starting point with low energy and no obvious symmetries.
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add a tiny bit of noise to break any residual symmetries or numerical stagnation
    np.random.seed(42) # Fixed seed for reproducibility
    x0 = x0 + 0.001 * np.random.randn(N, 3)
    # Normalize back to unit sphere
    norms = np.linalg.norm(x0, axis=1, keepdims=True)
    x0 = x0 / norms

    # 2. Define Objective Function and Gradient
    # Energy: sum_{i<j} log(1 / ||x_i - x_j||) = - sum_{i<j} log(||x_i - x_j||)
    # Penalty: lambda * sum (||x_i||^2 - 1)^2 to keep points on the sphere.
    
    lambda_pen = 10000.0
    
    def energy_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        
        # Compute pairwise distances
        # To avoid division by zero, we can add a small epsilon, but points shouldn't be identical.
        # Using broadcasting for efficiency
        # dist2[i, j] = ||x_i - x_j||^2
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        
        # Mask diagonal to avoid log(0)
        mask = ~np.eye(N, dtype=bool)
        dist2_masked = np.where(mask, dist2, 1.0)
        
        # Log energy
        # E = -0.5 * sum_{i != j} log(sqrt(dist2)) = -0.25 * sum_{i != j} log(dist2)
        # Wait, sum_{i<j} log(1/d) = - sum_{i<j} log(d).
        # sum_{i != j} log(d) = 2 * sum_{i<j} log(d).
        # So E = -0.5 * sum_{i != j} log(sqrt(dist2)) = -0.25 * sum_{i != j} log(dist2).
        
        log_dist2 = np.log(dist2_masked)
        e_log = -0.25 * np.sum(log_dist2[mask])
        
        # Penalty
        norms2 = np.sum(x ** 2, axis=1)
        pen = lambda_pen * np.sum((norms2 - 1.0) ** 2)
        
        total_energy = e_log + pen
        
        # Gradient
        # Grad of e_log w.r.t x_k:
        # E = - sum_{i<j} log(||x_i - x_j||)
        # dE/dx_k = sum_{j != k} (x_k - x_j) / ||x_k - x_j||^2
        # This is equivalent to sum_{j != k} diff[k, j] / dist2[k, j]
        
        # Avoid division by zero on diagonal
        dist2_safe = np.where(mask, dist2, 1.0)
        
        # grad_log[k, j] = diff[k, j] / dist2_safe[k, j]
        # We only sum over j != k, which is handled by the mask in the sum.
        # However, the formula sum_{j != k} (x_k - x_j)/dist2 is exactly what we want.
        # Note: diff[k, j] = x_k - x_j.
        
        grad_log = diff / dist2_safe[:, :, np.newaxis]
        # Sum over j (axis 1) for each k
        grad_log = np.sum(grad_log, axis=1)
        
        # The factor from the -0.25 in energy?
        # E = -0.25 * sum_{i!=j} log(dist2_ij)
        # dE/dx_k = -0.25 * sum_{j!=k} (1/dist2_kj) * d(dist2_kj)/dx_k
        # d(dist2_kj)/dx_k = 2 * (x_k - x_j)
        # So term is -0.25 * 2 * (x_k - x_j) / dist2_kj = -0.5 * (x_k - x_j) / dist2_kj
        # My grad_log calculation above was just (x_k - x_j) / dist2.
        # So I need to multiply by -0.5.
        
        grad_e = -0.5 * grad_log
        
        # Grad of penalty
        # P = lambda * sum (||x_i||^2 - 1)^2
        # dP/dx_k = lambda * 2 * (||x_k||^2 - 1) * 2 * x_k = 4 * lambda * (||x_k||^2 - 1) * x_k
        grad_pen = 4.0 * lambda_pen * (norms2 - 1.0)[:, np.newaxis] * x
        
        total_grad = grad_e + grad_pen
        
        return total_energy, total_grad.flatten()

    # 3. Optimization
    # Use L-BFGS-B with analytic gradient
    res = minimize(
        energy_and_grad, 
        x0.flatten(), 
        method='L-BFGS-B', 
        jac=True, 
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    x_opt = res.x.reshape(N, 3)
    
    # 4. Final Projection
    # Ensure points are exactly on the unit sphere
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    # Avoid division by zero if any point collapsed (unlikely with penalty)
    norms = np.where(norms == 0, 1.0, norms)
    x_final = x_opt / norms
    
    return x_final

# EVOLVE-BLOCK-END
