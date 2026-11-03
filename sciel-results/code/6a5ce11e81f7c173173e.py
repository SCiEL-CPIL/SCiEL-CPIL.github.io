import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _log_energy(X):
    """
    Compute the logarithmic energy.
    X: (N, 3) array of points.
    """
    # Compute pairwise distances
    # dist[i, j] = ||X[i] - X[j]||
    # To avoid zero division, we ensure points are distinct.
    # Using broadcasting
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid log(0)
    # If any distance is very small, energy is very high (negative log is large positive? No, log(1/d) = -log(d). If d->0, log(d)->-inf, so -log(d)->inf).
    # We want to minimize sum log(1/d).
    # The prompt says: "Points closer than 1e-9 ... make the candidate invalid."
    # So we should avoid getting that close.
    
    # Mask the diagonal
    mask = ~np.eye(N, dtype=bool)
    dist = dist[mask]
    
    # If any distance is 0, return a large penalty
    if np.any(dist < 1e-9):
        return 1e10
    
    energy = np.sum(np.log(1.0 / dist))
    return energy

def _gradient_energy(X):
    """
    Compute the gradient of the energy with respect to X.
    E = sum_{i<j} log(1/||x_i - x_j||) = - sum_{i<j} log(||x_i - x_j||)
    dE/dx_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero
    dist_sq[dist_sq < 1e-12] = 1e-12
    
    # Gradient term: (x_i - x_j) / dist^2
    # grad[i, j, k] = diff[i, j, k] / dist_sq[i, j]
    grad = diff / dist_sq[:, :, np.newaxis]
    
    # Sum over j for each i
    # The term for x_i in the sum over pairs (i,j) and (j,i)
    # E = sum_{i<j} ...
    # dE/dx_i gets contribution from pairs (i,j) where i<j and (j,i) where j<i?
    # Let's check the derivative carefully.
    # E = 1/2 sum_{i != j} log(1/||x_i - x_j||) = -1/2 sum_{i != j} log(||x_i - x_j||)
    # dE/dx_i = -1/2 sum_{j != i} (1/||x_i - x_j||) * (x_i - x_j)/||x_i - x_j||
    # = -1/2 sum_{j != i} (x_i - x_j)/||x_i - x_j||^2
    
    # My previous formula was sum_{j != i} (x_i - x_j)/||x_i - x_j||^2.
    # The factor of 1/2 is missing if I sum over all j != i.
    # If I sum over all j != i, I am counting each pair twice in the gradient sum?
    # No, the gradient of a term f(x_i, x_j) with respect to x_i is independent of the gradient with respect to x_j.
    # The sum is over all pairs i<j.
    # Let's use the symmetric form:
    # grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # This is the gradient of sum_{j != i} log(1/||x_i - x_j||).
    # Since E = 1/2 sum_{i != j} log(1/||x_i - x_j||), the gradient is 1/2 of that sum.
    
    grad_sum = np.sum(grad, axis=1)
    return 0.5 * grad_sum

def _project_to_sphere(X):
    """
    Project points to the unit sphere.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-12] = 1e-12
    return X / norms

def _fibonacci_sphere(N):
    """
    Generate Fibonacci sphere points.
    """
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(N):
    """
    Generate random points on the sphere.
    """
    v = np.random.randn(N, 3)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def _minimize_energy(X_init, time_limit=10.0):
    """
    Perform gradient descent with projection to minimize energy.
    """
    X = X_init.copy()
    best_X = X.copy()
    best_E = _log_energy(X)
    
    start_time = time.time()
    alpha = 0.01
    min_alpha = 1e-6
    
    while time.time() - start_time < time_limit:
        G = _gradient_energy(X)
        
        # Check for NaNs
        if not np.all(np.isfinite(G)):
            break
            
        # Update
        X_new = X - alpha * G
        X_new = _project_to_sphere(X_new)
        
        E_new = _log_energy(X_new)
        
        if not np.isfinite(E_new) or E_new > 1e9:
            # Step was too large or caused singularity
            alpha *= 0.5
            if alpha < min_alpha:
                break
            continue
            
        if E_new < best_E:
            best_E = E_new
            best_X = X_new.copy()
            
        # Simple adaptive step size
        # If energy decreased, maybe increase alpha slightly?
        # If energy increased, decrease alpha.
        # Here we only update best if it's lower.
        # Let's decrease alpha slowly to ensure convergence
        alpha *= 0.99
        
        X = X_new
        
    return best_X, best_E

def construct():
    """
    Construct the optimal configuration.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_X = None
    best_E = np.inf
    
    # Try Fibonacci first
    X_fib = _fibonacci_sphere(N)
    X_opt, E_opt = _minimize_energy(X_fib, time_limit=10.0)
    if E_opt < best_E:
        best_E = E_opt
        best_X = X_opt
        
    # Try a few random restarts
    # How many can we do?
    # Each restart takes ~10s. We have ~100s left. So ~10 restarts.
    num_restarts = 10
    
    for i in range(num_restarts):
        if time.time() - start_time > deadline:
            break
            
        # Generate a random start, or a perturbed Fibonacci
        # Perturbed Fibonacci might be better to stay in a good basin
        # Let's mix: some random, some perturbed fib
        
        if i % 2 == 0:
            X_init = _random_sphere(N)
        else:
            X_init = _fibonacci_sphere(N)
            # Add noise
            noise = np.random.randn(N, 3) * 0.1
            X_init = X_init + noise
            X_init = _project_to_sphere(X_init)
            
        X_opt, E_opt = _minimize_energy(X_init, time_limit=8.0)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
            
    # Final polish?
    # If we have time, run one more iteration with smaller step size on the best found
    if time.time() - start_time < deadline - 5.0 and best_X is not None:
        X_opt, E_opt = _minimize_energy(best_X, time_limit=5.0)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
            
    if best_X is None:
        # Fallback
        best_X = _fibonacci_sphere(N)
        
    return best_X
