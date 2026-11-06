"""Rewritten program: Riemannian gradient descent + L-BFGS polish for log-energy on S^2."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
TIME_LIMIT = 115.0  # safety margin before 120s kill

def _energy_and_grad(X):
    """Compute log-energy and its gradient for points X on unit sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Uses chordal distance.
    """
    # Pairwise dot products
    D = X @ X.T  # (N, N)
    # Chordal distance squared: |x_i - x_j|^2 = 2 - 2*x_i.x_j
    d2 = 2.0 - 2.0 * D
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(np.maximum(d2, 1e-15))
    
    # Mask for i < j
    iu = np.triu_indices(N, 1)
    d_ij = d[iu]
    
    # Energy: sum log(1/d) = -sum log(d)
    log_d = np.log(d_ij)
    energy = -np.sum(log_d)
    
    # Gradient: d/dx_i [ -log|d_ij| ] for pair (i,j)
    # d/dx_i log|d_ij| = (x_i - x_j) / |d_ij|^2 * (-1) ... let's derive carefully
    # E = -sum_{i<j} log|d_ij|, d_ij = |x_i - x_j|
    # dE/dx_i = -sum_{j>i} d/dx_i log|d_ij| - sum_{j<i} d/dx_i log|d_ji|
    # d/dx_i log|d_ij| = (x_i - x_j) / |d_ij|^2
    # So dE/dx_i = -sum_{j>i} (x_i - x_j)/d_ij^2 - sum_{j<i} (x_i - x_j)/d_ji^2
    #            = sum_{j>i} (x_j - x_i)/d_ij^2 + sum_{j<i} (x_j - x_i)/d_ji^2
    #            = sum_{j != i} (x_j - x_i) / d_ij^2
    
    # grad[i] = sum_{j != i} (x_j - x_i) / d_ij^2
    # Note: d_ij^2 = d2[i,j]
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j (x_j - x_i) * inv_d2[i,j] = (X.T @ inv_d2) - x_i * sum_j inv_d2[i,j]
    # Let's compute: sum_j x_j * inv_d2[i,j] = (inv_d2 @ X)[i]  -- wait, inv_d2 is (i,j), X is (j,3)
    # (inv_d2 @ X)[i] = sum_j inv_d2[i,j] * X[j]
    # sum_j inv_d2[i,j] = inv_d2[i].sum()
    grad = inv_d2 @ X - X * inv_d2.sum(axis=1, keepdims=True)
    
    return energy, grad


def _project(X):
    """Project points to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _fibonacci_sphere(n, seed=0):
    """Generate Fibonacci sphere points with optional jitter."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if seed > 0:
        # Add small perturbation
        X += rng.randn(n, 3) * 0.01
        X = _project(X)
    return X


def _lbfgs_polish(X, max_iter=500, tol=1e-10):
    """Polish using L-BFGS-B on tangent space (or just Euclidean with projection)."""
    # We'll use Euclidean L-BFGS with projection at each step
    def fun_g(x):
        Xp = x.reshape(N, 3)
        Xn = _project(Xp)
        e, g = _energy_and_grad(Xn)
        # Project gradient to tangent space: g - (g.x)x
        g_tangent = g - (g * Xn).sum(axis=1, keepdims=True) * Xn
        return e, g_tangent.reshape(-1)
    
    x0 = X.flatten()
    res = minimize(fun_g, x0, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50})
    return _project(res.x.reshape(N, 3))


def construct():
    start_time = time.time()
    
    # Multiple initializations
    best_X = None
    best_E = np.inf
    
    # Initializations: Fibonacci with different jitter seeds
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    
    for seed in seeds:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        X = _fibonacci_sphere(N, seed=seed)
        
        # Phase 1: Riemannian gradient descent with adaptive step size
        lr = 0.05
        min_lr = 1e-6
        max_iter = 3000
        patience = 0
        best_E_local = _energy_and_grad(X)[0]
        best_X_local = X.copy()
        
        for it in range(max_iter):
            if time.time() - start_time > TIME_LIMIT - 20:
                break
            
            E, g = _energy_and_grad(X)
            
            # Project gradient to tangent space
            g_tan = g - (g * X).sum(axis=1, keepdims=True) * X
            
            # Adaptive step: reduce if energy increases
            X_new = _project(X + lr * g_tan)
            E_new = _energy_and_grad(X_new)[0]
            
            if E_new < E:
                X = X_new
                patience = 0
                # Slightly increase lr if improving fast
                if patience == 0 and it > 100:
                    lr = min(lr * 1.02, 0.1)
            else:
                patience += 1
                lr *= 0.5
                if lr < min_lr:
                    break
            
            # Track best
            if E_new < best_E_local:
                best_E_local = E_new
                best_X_local = X_new.copy()
        
        X = best_X_local
        E = best_E_local
        
        # Phase 2: L-BFGS polish
        if time.time() - start_time < TIME_LIMIT - 15:
            X = _lbfgs_polish(X, max_iter=400, tol=1e-10)
            E = _energy_and_grad(X)[0]
        
        # Phase 3: Fine polish with many tiny steps
        if time.time() - start_time < TIME_LIMIT - 5:
            lr = 1e-5
            for it in range(2000):
                if time.time() - start_time > TIME_LIMIT - 3:
                    break
                E, g = _energy_and_grad(X)
                g_tan = g - (g * X).sum(axis=1, keepdims=True) * X
                X = _project(X + lr * g_tan)
                lr *= 0.999  # slow decay
        
        E_final = _energy_and_grad(X)[0]
        if E_final < best_E:
            best_E = E_final
            best_X = X.copy()
    
    # Final safety: ensure valid output
    if best_X is None:
        best_X = _fibonacci_sphere(N, seed=0)
    
    # Verify no points too close
    D = best_X @ best_X.T
    d2 = 2.0 - 2.0 * D
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    iu = np.triu_indices(N, 1)
    min_d = d[iu].min()
    if min_d < 1e-9:
        # Fallback
        best_X = _fibonacci_sphere(N, seed=0)
    
    # Ensure finite
    if not np.all(np.isfinite(best_X)):
        best_X = _fibonacci_sphere(N, seed=0)
    
    return best_X
