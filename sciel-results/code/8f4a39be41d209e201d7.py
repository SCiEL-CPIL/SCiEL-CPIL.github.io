"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import math

N = 150

# EVOLVE-BLOCK-START
def _log_energy(X):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = X.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X[i] - X[j])
            if d < 1e-9:
                return -1e18
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_fast(X):
    """Vectorized log energy for speed (uses sum, not fsum)."""
    n = X.shape[0]
    # Compute pairwise distances
    # X is (n, 3)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    dists = dists[mask]
    # Avoid log(0)
    dists = np.maximum(dists, 1e-15)
    return -np.sum(np.log(dists)) / 2.0  # Wait, sum over i<j means each pair once.
    # Actually: sum_{i<j} log(1/d_{ij}) = -sum_{i<j} log(d_{ij})
    # The mask gives upper triangle (n*(n-1)/2 elements).
    # Let's recompute properly.
    triu_indices = np.triu_indices(n, k=1)
    d = dists[triu_indices[0], triu_indices[1]]
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))


def _log_energy_grad(X):
    """
    Compute log energy and its gradient.
    Energy E = sum_{i<j} log(1/d_{ij}) = -sum_{i<j} log(d_{ij})
    d_{ij} = ||x_i - x_j||
    dE/dx_i = -sum_{j != i} (1/d_{ij}) * (x_i - x_j)/d_{ij} = -sum_{j != i} (x_i - x_j)/d_{ij}^2
    Wait: d/dx_i log(d_{ij}) = (1/d_{ij}) * (x_i - x_j)/d_{ij} = (x_i - x_j)/d_{ij}^2
    So dE/dx_i = -sum_{j != i} (x_i - x_j)/d_{ij}^2
    """
    n = X.shape[0]
    # Compute pairwise differences
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, 3)
    dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (n, n)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # E = -sum_{i<j} log(d_{ij})
    triu_i, triu_j = np.triu_indices(n, k=1)
    d = dists[triu_i, triu_j]
    d = np.maximum(d, 1e-15)
    E = -np.sum(np.log(d))
    
    # Gradient
    # grad_i = -sum_{j != i} (x_i - x_j) / d_{ij}^2
    # Let's compute a matrix W where W_{ij} = 1/d_{ij}^2 for i != j, 0 otherwise
    W = 1.0 / (dists ** 2)
    np.fill_diagonal(W, 0.0)
    
    # grad = -sum_j W_{ij} (x_i - x_j) = -sum_j W_{ij} x_i + sum_j W_{ij} x_j
    # = -x_i * sum_j W_{ij} + (W @ X)_i
    row_sums = np.sum(W, axis=1)  # (n,)
    WX = W @ X  # (n, 3)
    grad = -X * row_sums[:, np.newaxis] + WX
    
    return E, grad


def _project_to_sphere(X):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configuration: Fibonacci sphere
    X = _fibonacci_sphere(N)
    X = _project_to_sphere(X)
    
    # Compute initial energy
    E_curr = _log_energy_fast(X)
    X_best = X.copy()
    E_best = E_curr
    
    # Gradient descent with adaptive step size
    # Use L-BFGS style or simple gradient descent with line search
    # For speed, use simple gradient descent with decreasing step size
    
    lr = 0.01
    patience = 0
    max_patience = 500
    min_lr = 1e-10
    
    while time.time() < deadline:
        E, grad = _log_energy_grad(X)
        
        # Check if we've improved
        if E < E_best - 1e-12:
            E_best = E
            X_best = X.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            # Try to restart with a different initialization
            # Or reduce learning rate further
            if lr > min_lr:
                lr *= 0.5
                patience = 0
                continue
            else:
                break
        
        # Gradient step
        X_new = X - lr * grad
        X_new = _project_to_sphere(X_new)
        
        # Simple line search: check if energy decreased
        E_new = _log_energy_fast(X_new)
        
        if E_new < E:
            X = X_new
            # Maybe increase lr slightly if we're making good progress
            if patience == 0 and lr < 0.1:
                lr *= 1.1
        else:
            # Decrease lr
            lr *= 0.5
            patience += 1
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-10:
            break
    
    # Final polish: use scipy.optimize.minimize for a few iterations
    if time.time() < deadline - 10.0:
        try:
            from scipy.optimize import minimize
            
            def objective(x_flat):
                X_local = x_flat.reshape(N, 3)
                X_local = _project_to_sphere(X_local)
                E_local, _ = _log_energy_grad(X_local)
                return E_local
            
            def gradient(x_flat):
                X_local = x_flat.reshape(N, 3)
                X_local = _project_to_sphere(X_local)
                _, grad_local = _log_energy_grad(X_local)
                return grad_local.flatten()
            
            x0 = X_best.flatten()
            res = minimize(objective, x0, jac=gradient, method='BFGS', 
                          options={'maxiter': 100, 'gtol': 1e-10})
            
            if res.success:
                X_res = _project_to_sphere(res.x.reshape(N, 3))
                E_res = _log_energy_fast(X_res)
                if E_res < E_best:
                    E_best = E_res
                    X_best = X_res
        except Exception:
            pass
    
    # Ensure the result is valid
    X_final = _project_to_sphere(X_best)
    
    # Check for any points too close together
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(X_final[i] - X_final[j])
            if d < 1e-9:
                # Nudge apart
                X_final[j] += 1e-8 * np.random.randn(3)
                X_final[j] = _project_to_sphere(X_final[j])
    
    return X_final
# EVOLVE-BLOCK-END
