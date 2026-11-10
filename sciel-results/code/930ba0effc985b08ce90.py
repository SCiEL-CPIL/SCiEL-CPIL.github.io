"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
import math

def _fibonacci_spiral(n=N):
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _project_to_sphere(X):
    """Project points to unit sphere (handles zero vectors safely)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _log_energy(X):
    """Compute log energy: sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = X.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X[i] - X[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _log_energy_fast(X):
    """Vectorized log energy for optimization (not for final scoring)."""
    n = X.shape[0]
    # Pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _log_energy_and_grad(X):
    """Compute log energy and gradient via vectorized computation."""
    n = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Gradient of log(1/d) = -1/d * (x_i - x_j)/d = -(x_i - x_j)/d^2
    # But we sum over i<j, so for each point i, grad_i = sum_{j!=i} (x_i - x_j)/d_ij^2
    # Wait: d/dx_i log(1/|x_i-x_j|) = d/dx_i (-log|x_i-x_j|) = -(x_i-x_j)/|x_i-x_j|^2
    # So gradient of energy w.r.t. x_i = -sum_{j!=i} (x_i-x_j)/d_ij^2
    
    # For efficiency, compute pairwise
    inv_d2 = np.zeros_like(dists)
    mask = dists > 1e-15
    inv_d2[mask] = 1.0 / (dists[mask] ** 2)
    inv_d2[~mask] = 0.0
    
    # grad[i] = -sum_j (x_i - x_j) / d_ij^2
    # = -sum_j x_i/d_ij^2 + sum_j x_j/d_ij^2
    # = -x_i * sum_j inv_d2[i,j] + sum_j x_j * inv_d2[i,j]
    
    sum_inv = np.sum(inv_d2, axis=1)  # shape (n,)
    # sum_j x_j * inv_d2[i,j] -> matmul
    sum_xj_inv = inv_d2 @ X  # shape (n, 3)
    
    grad = -X * sum_inv[:, np.newaxis] + sum_xj_inv
    # Each point appears in pairs with all others, but we counted both (i,j) and (j,i)
    # In the energy sum over i<j, each pair contributes to both i and j.
    # The gradient formula above already accounts for this correctly since
    # dE/dx_i = sum_{j>i} d/dx_i log(1/d_ij) + sum_{j<i} d/dx_i log(1/d_ji)
    # = sum_{j!=i} -(x_i - x_j)/d_ij^2
    # So grad is correct as computed.
    
    energy = _log_energy_fast(X)
    return energy, grad

def _project_and_normalize(X):
    """Project to sphere and ensure validity."""
    X = _project_to_sphere(X)
    return X

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 5s before 150s kill
    
    # Start with Fibonacci spiral
    X = _fibonacci_spiral(N)
    best_X = X.copy()
    best_energy = _log_energy_fast(X)
    
    # Use L-BFGS-B or similar optimizer
    from scipy.optimize import minimize
    
    def objective(x_flat):
        X = x_flat.reshape(N, 3)
        X = _project_to_sphere(X)
        return _log_energy_fast(X)
    
    def gradient(x_flat):
        X = x_flat.reshape(N, 3)
        X = _project_to_sphere(X)
        _, grad = _log_energy_and_grad(X)
        return grad.flatten()
    
    # Multiple restarts with different initial conditions
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            X0 = _fibonacci_spiral(N)
        elif restart == 1:
            # Slightly perturbed Fibonacci
            X0 = _fibonacci_spiral(N)
            X0 += 0.1 * np.random.randn(N, 3)
            X0 = _project_to_sphere(X0)
        elif restart == 2:
            # Different golden ratio variant
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + 0.25)
            r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
            X0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 3:
            # Random start
            X0 = np.random.randn(N, 3)
            X0 = _project_to_sphere(X0)
        else:
            # Another perturbed variant
            X0 = _fibonacci_spiral(N)
            X0 += 0.3 * np.random.randn(N, 3)
            X0 = _project_to_sphere(X0)
        
        x0 = X0.flatten()
        
        # Use L-BFGS-B with projection
        # We'll do manual projected gradient descent for more control
        
        # Actually, let's use a simple projected gradient method
        lr = 0.01
        best_local_energy = _log_energy_fast(X0)
        best_local_X = X0.copy()
        
        for step in range(5000):
            if time.time() > deadline - 10.0:
                break
            
            X = _project_to_sphere(x0.reshape(N, 3))
            energy, grad = _log_energy_and_grad(X)
            
            # Project gradient to tangent space
            # Tangent space: vectors orthogonal to X
            grad_tangent = grad - np.sum(grad * X, axis=1, keepdims=True) * X
            
            # Adaptive step size
            x_new = X - lr * grad_tangent
            x_new = _project_to_sphere(x_new)
            
            e_new = _log_energy_fast(x_new)
            
            if e_new < best_local_energy:
                best_local_energy = e_new
                best_local_X = x_new.copy()
                # Increase learning rate slightly
                lr = min(lr * 1.01, 0.05)
            else:
                # Decrease learning rate
                lr *= 0.9
                if lr < 1e-6:
                    break
            
            x0 = x_new.flatten()
        
        if best_local_energy < best_energy:
            best_energy = best_local_energy
            best_X = best_local_X.copy()
    
    # Final polish with higher precision using math.fsum-based evaluation
    # Do a few more gradient steps with very small lr
    lr = 1e-5
    X = best_X.copy()
    for step in range(2000):
        if time.time() > deadline - 5.0:
            break
        
        energy, grad = _log_energy_and_grad(X)
        grad_tangent = grad - np.sum(grad * X, axis=1, keepdims=True) * X
        X_new = _project_to_sphere(X - lr * grad_tangent)
        
        e_new = _log_energy_fast(X_new)
        if e_new < energy:
            X = X_new
        else:
            lr *= 0.5
            if lr < 1e-10:
                break
    
    # Ensure final result is valid
    X_final = _project_to_sphere(X)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(X_final[i] - X_final[j])
            if d < 1e-9:
                # Perturb slightly
                X_final[j] += 1e-8 * np.random.randn(3)
                X_final = _project_to_sphere(X_final)
                break
    
    return X_final
# EVOLVE-BLOCK-END
