import numpy as np
import time
import math

N = 150

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s buffer before 150s kill
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def energy_and_grad(X):
        """
        Computes the logarithmic energy and its gradient.
        E = sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
        Grad_i = sum_{j!=i} (x_j - x_i) / ||x_i - x_j||^2
        """
        # Pairwise differences
        # X shape (N, 3)
        # D shape (N, N, 3)
        D = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        
        # Distances
        # dist shape (N, N)
        dist = np.linalg.norm(D, axis=2)
        
        # Mask diagonal to avoid division by zero
        mask = ~np.eye(N, dtype=bool)
        
        # Energy
        # We sum log(1/d) for i<j
        # Sum over all i != j of -log(d_ij) is 2 * E
        # So E = 0.5 * sum_{i!=j} -log(d_ij)
        # log(1/d) = -log(d)
        with np.errstate(divide='ignore'):
            log_d = np.log(dist)
        log_d[~mask] = 0.0 # Ignore diagonal
        
        E = -0.5 * np.sum(log_d)
        
        # Gradient
        # grad_i = sum_{j!=i} (x_j - x_i) / d_ij^2
        # Note: D[i,j] = x_i - x_j. So x_j - x_i = -D[i,j].
        # Term is -D[i,j] / dist[i,j]^2
        # dist_sq = dist^2
        dist_sq = dist**2
        dist_sq[~mask] = 1.0 # Avoid div by zero, will be masked later
        
        # grad_contrib[i,j] = (x_j - x_i) / d_ij^2 = -D[i,j] / dist_sq[i,j]
        # Shape (N, N, 3)
        grad_contrib = -D / dist_sq[:, :, np.newaxis]
        
        # Sum over j (axis 1)
        Grad = np.sum(grad_contrib, axis=1)
        
        # Zero out diagonal contributions (already handled by mask in dist_sq? 
        # Wait, dist_sq[~mask]=1.0, so D[i,i]=0, so grad_contrib[i,i] = 0.
        # So diagonal is 0. Good.)
        
        return E, Grad

    # 2. Optimization: Riemannian Gradient Descent
    E, Grad = energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    lr = 0.1 # Initial learning rate
    min_lr = 1e-8
    
    while time.time() < deadline:
        # Project gradient to tangent space
        # g_tan = g - (g . x) x
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Grad_tan = Grad - proj * X
        
        # Step
        X_new = X + lr * Grad_tan
        
        # Normalize
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        X_new = X_new / norms
        
        # Compute new energy
        E_new, Grad_new = energy_and_grad(X_new)
        
        if E_new < best_E:
            best_E = E_new
            best_X = X_new.copy()
        
        if E_new < E:
            # Accept step
            X = X_new
            E = E_new
            Grad = Grad_new
            # Increase learning rate slightly if progress
            lr = min(lr * 1.05, 1.0)
        else:
            # Reject step, reduce learning rate
            lr *= 0.5
            if lr < min_lr:
                # Try a few random perturbations to escape local minima?
                # For now, just break if stuck
                # Or try a larger random kick
                break

    # Final check: ensure points are on unit sphere and valid
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    best_X = best_X / norms
    
    # Verify no points are too close
    D = best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]
    dist = np.linalg.norm(D, axis=2)
    np.fill_diagonal(dist, 1.0)
    if np.min(dist) < 1e-9:
        # Fallback to Fibonacci if collision (unlikely)
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    return best_X
