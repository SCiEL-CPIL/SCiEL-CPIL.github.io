import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    """
    Minimizes the logarithmic energy of N points on the unit sphere.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    X_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Flatten for scipy optimizer
    x0 = X_init.flatten()
    
    # Penalty weight for sphere constraint
    # The energy gradient can be large if points are close.
    # A large penalty ensures we stay on the sphere.
    lambda_pen = 1000.0
    
    def energy_and_grad(x_flat):
        X = x_flat.reshape(N, 3)
        
        # Calculate pairwise distances
        # dist[i, j] = ||X[i] - X[j]||
        # Using broadcasting
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Avoid division by zero (should not happen if points are distinct)
        dist_sq[dist_sq < 1e-12] = 1e-12
        dist = np.sqrt(dist_sq)
        
        # Logarithmic Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
        # We sum over all i != j and divide by 2
        # Mask out diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # Energy
        # log(1/d) = -log(d)
        log_dist = np.log(dist)
        E_log = -np.sum(log_dist[mask]) / 2.0
        
        # Penalty Energy
        norms_sq = np.sum(X * X, axis=1)
        E_pen = lambda_pen * np.sum((norms_sq - 1.0) ** 2)
        
        E_total = E_log + E_pen
        
        # Gradient
        # Gradient of E_log wrt X[k]
        # d/dx_k (-log(||x_k - x_j||)) = -(x_k - x_j) / ||x_k - x_j||^2
        # Sum over j != k
        # grad_log[k] = - sum_{j!=k} (X[k] - X[j]) / dist_sq[k,j]
        
        # We can compute this efficiently using matrix operations
        # Let M be a matrix where M[k, j] = 1/dist_sq[k, j] for k!=j, 0 otherwise
        # Then grad_log = - (M * (X[:, np.newaxis, :] - X[np.newaxis, :, :])) summed over j?
        # Actually, force on k from j is (X[k] - X[j]) / dist_sq[k, j].
        # Total force on k is sum_j (X[k] - X[j]) / dist_sq[k, j].
        
        inv_dist_sq = np.zeros_like(dist_sq)
        inv_dist_sq[mask] = 1.0 / dist_sq[mask]
        
        # diff is (N, N, 3). We want to sum over axis 1 (j)
        # grad_log[k] = sum_j inv_dist_sq[k, j] * (X[k] - X[j])
        # Note: diff[k, j] = X[k] - X[j]
        grad_log = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
        
        # Gradient of E_pen
        # d/dx_k (lambda * (||x_k||^2 - 1)^2) = lambda * 2 * (||x_k||^2 - 1) * 2 * X[k] = 4 * lambda * (||x_k||^2 - 1) * X[k]
        grad_pen = 4.0 * lambda_pen * (norms_sq - 1.0)[:, np.newaxis] * X
        
        grad_total = grad_log + grad_pen
        
        return E_total, grad_total
    
    # Optimization
    # We use L-BFGS-B as it is efficient for large N
    # We don't have bounds on coordinates directly, but the penalty handles the sphere.
    # However, to prevent points from flying away, we could add bounds [-1.1, 1.1]
    bounds = [(-1.1, 1.1)] * (N * 3)
    
    best_x = x0
    best_val = energy_and_grad(x0)[0]
    
    try:
        res = minimize(
            energy_and_grad, 
            x0, 
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
        )
        
        if time.time() < deadline:
            if res.success or res.fun < best_val:
                best_x = res.x
                best_val = res.fun
            
            # Second pass: Polish with tighter tolerance and higher penalty
            # Re-initialize penalty for polish? Or just run again.
            # Let's project first to be on sphere, then run a local Newton-like step or another L-BFGS
            X_best = best_x.reshape(N, 3)
            norms = np.linalg.norm(X_best, axis=1)
            X_best = X_best / norms[:, np.newaxis]
            
            # Run a short polish
            x_polish = X_best.flatten()
            
            # Use a higher penalty for the polish to ensure strict sphere constraint
            lambda_pen_polish = 10000.0
            
            def energy_and_grad_polish(x_flat):
                X = x_flat.reshape(N, 3)
                diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
                dist_sq = np.sum(diff * diff, axis=2)
                dist_sq[dist_sq < 1e-12] = 1e-12
                dist = np.sqrt(dist_sq)
                mask = ~np.eye(N, dtype=bool)
                
                log_dist = np.log(dist)
                E_log = -np.sum(log_dist[mask]) / 2.0
                
                norms_sq = np.sum(X * X, axis=1)
                E_pen = lambda_pen_polish * np.sum((norms_sq - 1.0) ** 2)
                
                E_total = E_log + E_pen
                
                inv_dist_sq = np.zeros_like(dist_sq)
                inv_dist_sq[mask] = 1.0 / dist_sq[mask]
                
                grad_log = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
                grad_pen = 4.0 * lambda_pen_polish * (norms_sq - 1.0)[:, np.newaxis] * X
                
                grad_total = grad_log + grad_pen
                
                return E_total, grad_total
            
            res2 = minimize(
                energy_and_grad_polish,
                x_polish,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            
            if time.time() < deadline:
                if res2.success or res2.fun < best_val:
                    best_x = res2.x
                    best_val = res2.fun
                    
    except Exception as e:
        # If optimization fails, return the initial or best found so far
        pass

    # Final projection to unit sphere
    X_final = best_x.reshape(N, 3)
    norms = np.linalg.norm(X_final, axis=1)
    # Handle zero norms (shouldn't happen)
    norms[norms < 1e-9] = 1.0
    X_final = X_final / norms[:, np.newaxis]
    
    return X_final
