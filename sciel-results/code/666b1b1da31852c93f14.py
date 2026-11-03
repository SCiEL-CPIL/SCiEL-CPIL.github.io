import numpy as np
from scipy.optimize import minimize
import time
import math

N = 150

def _energy_and_grad(X):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    dE/dxi = -sum_{j≠i} (xi-xj)/|xi-xj|^2
    """
    n = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        grad_term = diff / (dist_sq[:, :, np.newaxis] + 1e-30)
    np.fill_diagonal(grad_term[:, :, 0], 0)
    np.fill_diagonal(grad_term[:, :, 1], 0)
    np.fill_diagonal(grad_term[:, :, 2], 0)
    
    grad = -np.sum(grad_term, axis=1)
    
    return energy, grad

def _energy_only(X):
    """Compute only the logarithmic energy."""
    n = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))

def _project_to_sphere(X):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return X / norms

def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate n random points on unit sphere."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    return _project_to_sphere(X)

def _lbfgs_optimize(X0, maxiter=1000, tol=1e-12):
    """Optimize using L-BFGS-B with sphere projection."""
    def objective_and_grad(x):
        X = x.reshape(-1, 3)
        X = _project_to_sphere(X)
        e, g = _energy_and_grad(X)
        # Gradient in the flattened space needs to account for projection
        # For simplicity, we'll use the gradient directly and let L-BFGS handle it
        return e, g.flatten()
    
    x0 = X0.flatten()
    res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                   jac=True, options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol})
    X_opt = _project_to_sphere(res.x.reshape(-1, 3))
    return X_opt, res.fun

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Try multiple initial configurations
    best_X = None
    best_energy = np.inf
    
    initializations = []
    
    # Fibonacci sphere
    initializations.append(_fibonacci_sphere(N))
    
    # Random points with different seeds
    for seed in [42, 123, 456, 789, 101]:
        if time.time() > deadline:
            break
        initializations.append(_random_sphere(N, seed=seed))
    
    for idx, X0 in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # Check if we have time for this optimization
        time_per_opt = 10.0  # Allow ~10s per optimization
        if time.time() + time_per_opt > deadline:
            break
        
        try:
            X_opt, energy = _lbfgs_optimize(X0, maxiter=500, tol=1e-10)
            
            if energy < best_energy:
                best_energy = energy
                best_X = X_opt.copy()
        except Exception:
            continue
    
    # If we have time, do a final polish on the best result
    if best_X is not None and time.time() < deadline - 5:
        try:
            X_polished, energy_polished = _lbfgs_optimize(best_X, maxiter=200, tol=1e-12)
            if energy_polished < best_energy:
                best_energy = energy_polished
                best_X = X_polished
        except Exception:
            pass
    
    # Ensure output is valid
    if best_X is None:
        # Fallback to Fibonacci sphere if optimization failed
        best_X = _fibonacci_sphere(N)
    
    result = _project_to_sphere(best_X.copy())
    
    # Check for very close points and separate them if needed
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(result[i] - result[j])
            if d < 1e-9:
                result[i] = result[i] + 1e-8 * np.random.randn(3)
                result[i] = result[i] / np.linalg.norm(result[i])
    
    return result
