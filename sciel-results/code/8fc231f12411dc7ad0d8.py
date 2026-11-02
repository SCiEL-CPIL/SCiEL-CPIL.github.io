import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Initialize
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Flatten
    x0 = X0.flatten()
    
    # Objective function
    def objective(x):
        X = x.reshape(N, 3)
        # Compute energy
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq[dist_sq < 1e-12] = 1e-12
        # E = sum_{i<j} log(1/sqrt(dist_sq)) = -0.5 * sum_{i<j} log(dist_sq)
        # sum_{i<j} is half of sum_{i!=j}
        # log(dist_sq) matrix
        log_dist_sq = np.log(dist_sq)
        np.fill_diagonal(log_dist_sq, 0.0)
        energy = -0.25 * np.sum(log_dist_sq)
        
        # Penalty for sphere constraint
        norms_sq = np.sum(X**2, axis=1)
        penalty = 100.0 * np.sum((norms_sq - 1.0)**2)
        
        return energy + penalty

    # Gradient
    def gradient(x):
        X = x.reshape(N, 3)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq[dist_sq < 1e-12] = 1e-12
        
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # Gradient of energy
        grad_energy = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Gradient of penalty
        # P = 100 * sum (norm^2 - 1)^2
        # dP/dx_i = 100 * 2 * (norm^2 - 1) * 2 * x_i = 400 * (norm^2 - 1) * x_i
        norms_sq = np.sum(X**2, axis=1, keepdims=True)
        grad_penalty = 400.0 * (norms_sq - 1.0) * X
        
        grad = grad_energy + grad_penalty
        return grad.flatten()

    # Run optimization
    # L-BFGS-B is efficient
    # We might need to run it multiple times or with high maxiter
    result = minimize(objective, x0, method='L-BFGS-B', jac=gradient, 
                      options={'maxiter': 500, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if time.time() > deadline - 10.0:
        # If time is running out, return what we have
        X = result.x.reshape(N, 3)
    else:
        # Try to polish with a second run or higher precision
        # Or just return
        X = result.x.reshape(N, 3)
        
    # Final projection to ensure exact unit length
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    
    return X
