import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def construct():
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    def energy_and_grad(y):
        # y is flat vector of size 3*N
        X = y.reshape(N, 3)
        
        # Normalize to ensure points are on sphere for evaluation
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        X = X / norms
        
        # Compute pairwise distances
        # diff[i, j] = X[i] - X[j]
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal to avoid log(0)
        mask = ~np.eye(N, dtype=bool)
        
        # Energy: sum log(1/d) = -sum log(d)
        # We only need upper triangle for the value, but gradient needs all
        # For value:
        # d_upper = dists[mask & (np.triu(np.ones((N,N), dtype=bool), 1))]
        # Actually, simpler:
        # sum over i<j of -log(d_ij)
        
        # For gradient, we need sum over j != i of (X_i - X_j) / d_ij^2
        # Let's compute the gradient matrix G where G[i] = - sum_{j!=i} (X_i - X_j) / d_ij^2
        
        # dists^2
        dists_sq = dists ** 2
        # Avoid division by zero on diagonal
        dists_sq[~mask] = 1.0 # Diagonal doesn't contribute to gradient sum anyway if we mask
        
        # We need to sum (X_i - X_j) / d_ij^2 over j
        # term[i, j, k] = (X[i, k] - X[j, k]) / d_ij^2
        # This is large (N*N*3). N=100, so 10,000*3 = 30,000 ops. Very fast.
        
        # Compute 1/d^2 for all pairs
        inv_d2 = np.zeros((N, N))
        inv_d2[mask] = 1.0 / dists_sq[mask]
        
        # Gradient w.r.t X_i:
        # grad_i = - sum_j (X_i - X_j) * inv_d2[i, j]
        # Note: inv_d2 is symmetric.
        # sum_j X_j * inv_d2[i, j] is a matrix multiplication: inv_d2 @ X
        # sum_j X_i * inv_d2[i, j] is X_i * sum_j inv_d2[i, j]
        
        sum_inv_d2 = np.sum(inv_d2, axis=1) # Shape (N,)
        
        # grad_ambient = - ( - X * sum_inv_d2[:, np.newaxis] + inv_d2 @ X )
        # = X * sum_inv_d2[:, np.newaxis] - inv_d2 @ X
        
        grad_ambient = X * sum_inv_d2[:, np.newaxis] - (inv_d2 @ X)
        
        # Project gradient onto tangent space
        # g_tangent = g - (g . x) x
        dot_products = np.sum(grad_ambient * X, axis=1) # Shape (N,)
        grad_tangent = grad_ambient - dot_products[:, np.newaxis] * X
        
        # Compute energy value for the optimizer
        # We need sum_{i<j} -log(d_ij)
        # Use only upper triangle to avoid double counting
        triu_mask = np.triu(np.ones((N, N), dtype=bool), 1)
        d_upper = dists[triu_mask]
        
        # Add small epsilon to avoid log(0) if points are extremely close
        d_upper = np.maximum(d_upper, 1e-12)
        
        # Energy is sum of log(1/d) = -log(d)
        energy = -np.sum(np.log(d_upper))
        
        return energy, grad_tangent.flatten()

    # Run L-BFGS-B
    # We pass the objective as a tuple (f, g)
    # L-BFGS-B is good for this size
    res = minimize(
        lambda y: energy_and_grad(y)[0],
        X0.flatten(),
        jac=lambda y: energy_and_grad(y)[1],
        method='L-BFGS-B',
        options={'maxiter': 2000, 'gtol': 1e-8, 'maxfun': 2000, 'disp': False}
    )
    
    # Check time
    if time.time() - start_time > time_limit:
         # Return the best found so far
         # res.x might be partial if it timed out? No, minimize returns when done or maxiter.
         # If we hit maxiter, it's the best it found.
         pass

    X_opt = res.x.reshape(N, 3)
    
    # Final normalization
    norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
    X_opt = X_opt / norms
    
    return X_opt
