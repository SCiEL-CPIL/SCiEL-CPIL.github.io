"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 145.0  # safety margin before 150s kill


def _log_energy(X):
    """Compute log energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = X.shape[0]
    # Compute all pairwise distances
    # Use squared distances to avoid sqrt, then sqrt
    # dist^2 = 2 - 2 * dot
    dots = X @ X.T
    np.fill_diagonal(dots, 1.0)
    d2 = 2.0 - 2.0 * dots
    np.maximum(d2, 1e-18, out=d2)  # avoid log(0)
    dist = np.sqrt(d2)
    # We need sum over i<j of log(1/dist) = -sum_{i<j} log(dist)
    # Get upper triangle indices
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Use math.fsum for high precision
    log_terms = -np.log(d)
    return math.fsum(log_terms.tolist())


def _log_energy_and_grad(X):
    """Compute log energy and gradient.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    Gradient w.r.t. x_i:
    d/dx_i [ -log(|x_i - x_j|) ] = -(x_i - x_j) / |x_i - x_j|^2
    
    For the full energy, grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = X.shape[0]
    dots = X @ X.T
    np.fill_diagonal(dots, 1.0)
    d2 = 2.0 - 2.0 * dots
    np.maximum(d2, 1e-18, out=d2)
    dist = np.sqrt(d2)
    
    # Gradient: grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # (x_i - x_j) = X[i] - X[j]
    # We can compute: for each i, sum over j of (X[i] - X[j]) / d2[i,j]
    # = X[i] * sum_j 1/d2[i,j] - sum_j X[j]/d2[i,j]
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)  # exclude self
    
    row_sum = inv_d2.sum(axis=1)  # shape (n,)
    # sum_j X[j] / d2[i,j] = inv_d2 @ X
    weighted_sum = inv_d2 @ X  # shape (n, 3)
    
    grad = X * row_sum[:, None] - weighted_sum
    
    # Energy
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    log_terms = -np.log(d)
    energy = math.fsum(log_terms.tolist())
    
    return energy, grad


def _project_to_sphere(X):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def construct():
    t_start = time.time()
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_X = X.copy()
    best_E = _log_energy(best_X)
    
    # Try multiple random restarts and gradient descent
    np.random.seed(42)
    
    # First, do gradient descent from the Fibonacci start
    # Use L-BFGS-B with sphere constraint handled by re-normalization
    # Actually, let's use a penalty-free approach: optimize on sphere directly
    
    def objective_and_grad(flat_x):
        X_local = flat_x.reshape(N, 3)
        E, G = _log_energy_and_grad(X_local)
        # Project gradient to tangent space: G - (G . x) * x
        grad_tangent = G - (G * X_local).sum(axis=1, keepdims=True) * X_local
        return E, grad_tangent.flatten()
    
    # Use L-BFGS-B with bounds, then project
    # Better: use a custom approach with re-normalization
    
    # Strategy: iterative gradient descent with re-normalization
    # At each step:
    # 1. Compute gradient on the sphere (tangent space)
    # 2. Take a step
    # 3. Re-normalize
    
    lr = 0.01
    for iteration in range(500):
        if time.time() > DEADLINE - 5:
            break
        
        E, G = _log_energy_and_grad(X)
        
        # Project to tangent space
        grad_tangent = G - (G * X).sum(axis=1, keepdims=True) * X
        
        # Adaptive learning rate
        X_new = X - lr * grad_tangent
        X_new = _project_to_sphere(X_new)
        
        E_new = _log_energy(X_new)
        
        if E_new < E:
            X = X_new
            if E_new < best_E:
                best_E = E_new
                best_X = X.copy()
            # Increase learning rate slightly
            lr = min(lr * 1.05, 0.1)
        else:
            lr = lr * 0.5
        
        if lr < 1e-6:
            break
    
    # Now try multiple random restarts with gradient descent
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > DEADLINE - 10:
            break
        
        # Random initialization
        X_rand = np.random.randn(N, 3)
        X_rand = _project_to_sphere(X_rand)
        
        # Gradient descent
        lr = 0.01
        for iteration in range(300):
            if time.time() > DEADLINE - 5:
                break
            
            E, G = _log_energy_and_grad(X_rand)
            
            # Project to tangent space
            grad_tangent = G - (G * X_rand).sum(axis=1, keepdims=True) * X_rand
            
            X_new = X_rand - lr * grad_tangent
            X_new = _project_to_sphere(X_new)
            
            E_new = _log_energy(X_new)
            
            if E_new < E:
                X_rand = X_new
                lr = min(lr * 1.05, 0.1)
            else:
                lr = lr * 0.5
            
            if lr < 1e-6:
                break
        
        E_final = _log_energy(X_rand)
        if E_final < best_E:
            best_E = E_final
            best_X = X_rand.copy()
    
    # Final polish: more iterations with smaller steps
    if time.time() < DEADLINE - 10:
        X = best_X.copy()
        lr = 0.001
        for iteration in range(200):
            if time.time() > DEADLINE - 2:
                break
            
            E, G = _log_energy_and_grad(X)
            grad_tangent = G - (G * X).sum(axis=1, keepdims=True) * X
            
            X_new = X - lr * grad_tangent
            X_new = _project_to_sphere(X_new)
            
            E_new = _log_energy(X_new)
            
            if E_new < E:
                X = X_new
                if E_new < best_E:
                    best_E = E_new
                    best_X = X.copy()
                lr = min(lr * 1.02, 0.01)
            else:
                lr = lr * 0.5
            
            if lr < 1e-8:
                break
    
    return best_X
