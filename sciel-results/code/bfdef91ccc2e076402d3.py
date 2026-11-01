"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def construct():
    # Start with a Fibonacci spiral as initial guess
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Use gradient-based optimization to minimize logarithmic energy
    # Energy: sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    # We minimize E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    
    def energy_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / norms
        
        # Compute pairwise distances
        # d_ij = ||xi - xj||
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Avoid log(0) and self-pairs
        np.fill_diagonal(dist, 1.0)
        
        # Energy = sum_{i<j} log(1/dist_ij) = -sum_{i<j} log(dist_ij)
        # Use only upper triangle
        iu = np.triu_indices(N, k=1)
        log_dists = np.log(dist[iu])
        energy = -np.sum(log_dists)
        
        # Gradient computation
        # dE/dx_k = sum_{j != k} dE/d(x_k - x_j)
        # E = -sum_{i<j} log(||xi - xj||)
        # dE/dx_k = sum_{j<k} (x_k - x_j)/||x_k - x_j||^2 * (-1) * ... 
        # Let's be more careful:
        # E = -sum_{i<j} log(||xi - xj||)
        # dE/dx_k = sum_{j: j<k} -d/dx_k log(||x_k - x_j||) + sum_{j: j>k} -d/dx_k log(||x_j - x_k||)
        # d/dx_k log(||x_k - x_j||) = (x_k - x_j)/||x_k - x_j||^2
        # d/dx_k log(||x_j - x_k||) = -(x_k - x_j)/||x_k - x_j||^2
        # So dE/dx_k = sum_{j<k} -(x_k - x_j)/||x_k - x_j||^2 + sum_{j>k} (x_k - x_j)/||x_k - x_j||^2
        # = sum_{j != k} (x_k - x_j)/||x_k - x_j||^2 * (sign depending on order)
        # Actually let me recompute:
        # For pair (i,j) with i<j, term is -log(||xi - xj||)
        # d/dxi: -(xi - xj)/||xi - xj||^2
        # d/dxj: +(xi - xj)/||xi - xj||^2 = -(xj - xi)/||xi - xj||^2
        # So for point k:
        # grad_k = sum_{j<k} -(x_k - x_j)/||x_k - x_j||^2 + sum_{j>k} -(x_k - x_j)/||x_k - x_j||^2
        # = -sum_{j != k} (x_k - x_j)/||x_k - x_j||^2
        
        grad = np.zeros_like(x)
        for k in range(N):
            for j in range(N):
                if j == k:
                    continue
                d = dist[k, j]
                if d < 1e-10:
                    continue
                grad[k] -= (x[k] - x[j]) / (d * d)
        
        # Project gradient onto tangent space of sphere
        # For unit sphere constraint, effective gradient is grad - (grad . x_k) * x_k
        for k in range(N):
            dot = np.dot(grad[k], x[k])
            grad[k] = grad[k] - dot * x[k]
        
        return energy, grad.flatten()
    
    # Use L-BFGS-B with many iterations
    res = minimize(
        lambda x: energy_and_grad(x)[0],
        x0.flatten(),
        jac=lambda x: energy_and_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 100000}
    )
    
    x_opt = res.x.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    x_opt = x_opt / norms
    
    return x_opt
# EVOLVE-BLOCK-END
