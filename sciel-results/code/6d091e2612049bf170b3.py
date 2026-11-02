"""Optimized logarithmic energy minimizer for 150 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150
DEADLINE = 110.0  # seconds, leaving margin before 120s kill

def _log_energy(X):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    # X is (N, 3)
    # Compute all pairwise distances
    # d_ij = ||x_i - x_j||
    # Using broadcasting
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    # Get upper triangle indices (i < j)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    # Clip to avoid log(0)
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))

def _log_energy_grad(X):
    """Compute energy and gradient."""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    dist = np.maximum(dist, 1e-15)
    
    # Energy
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: d/dx_i sum_{j!=i} log(1/|x_i - x_j|) = sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    # = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # But we need to be careful: log(1/d) = -log(d), so d(-log d)/dx = -1/d * dd/dx
    # dd/dx_i = (x_i - x_j)/d_ij
    # So d(energy)/dx_i = -sum_{j!=i} (1/d_ij) * (x_i - x_j)/d_ij = -sum_{j!=i} (x_i - x_j)/d_ij^2
    
    # Actually: energy = sum_{i<j} -log(d_ij)
    # d(energy)/d(x_i) = -sum_{j != i} d(-log d_ij)/d(x_i)
    # d(-log d_ij)/d(x_i) = -(1/d_ij) * d(d_ij)/d(x_i) = -(1/d_ij) * (x_i - x_j)/d_ij
    # So d(energy)/d(x_i) = -sum_{j != i} [-(x_i - x_j)/d_ij^2] = sum_{j != i} (x_i - x_j)/d_ij^2
    
    # For efficiency, use matrix form
    # grad[i] = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # = sum_j (x_i - x_j)/d_ij^2 - (x_i - x_i)/0  (exclude self)
    
    # Let's compute pairwise: A[i,j] = (x_i - x_j) / d_ij^2 for i != j
    # grad[i] = sum_{j != i} A[i,j]
    
    # d_ij^2
    dist_sq = dist * dist
    dist_sq = np.maximum(dist_sq, 1e-30)
    
    # A[i,j] = diff[i,j] / dist_sq[i,j]
    # But diff[i,i] = 0, so A[i,i] = 0/0 -> set to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        A = diff / dist_sq[:, :, np.newaxis]
    np.fill_diagonal(A, 0.0)
    
    grad = np.sum(A, axis=1)  # (N, 3)
    
    return energy, grad

def _project_to_sphere(X):
    """Project points to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _fibonacci_start():
    """Fibonacci spiral starting configuration."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_start(seed):
    """Random starting configuration."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, 3))
    return _project_to_sphere(X)

def _local_optimize(X0, time_limit):
    """Run L-BFGS-B optimization with time limit."""
    start_time = time.time()
    
    def obj_and_grad(flat):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        X = flat.reshape(N, 3)
        # Project to sphere for evaluation
        Xp = _project_to_sphere(X)
        e, g = _log_energy_grad(Xp)
        # Gradient w.r.t. flat parameterization: we need chain rule
        # Since we're projecting, the effective gradient is the tangential component
        # For L-BFGS-B on unconstrained parameters, we just use the projected gradient
        # Actually, better to use the gradient on the sphere projected back
        # g_tangent[i] = g[i] - (g[i].x_i) * x_i  (remove radial component)
        # But since we're optimizing in R^{3N} with projection, let's just return
        # the gradient as-is; L-BFGS-B will handle it, and we re-project each step
        return e, g.flatten()
    
    x0 = X0.flatten()
    
    result = minimize(
        obj_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
    )
    
    X_opt = _project_to_sphere(result.x.reshape(N, 3))
    return X_opt, result.fun

def _simulated_annealing(X0, time_limit, n_restarts=5):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    best_X = X0.copy()
    best_E = _log_energy(best_X)
    
    # Initial temperature
    T = 0.1
    T_final = 1e-6
    cooling = 0.999
    
    current_X = X0.copy()
    current_E = best_E
    
    max_iter = 50000
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        T = 0.1 * (0.5 ** restart)
        current_X = best_X.copy() if restart > 0 else X0.copy()
        current_E = best_E if restart > 0 else _log_energy(current_X)
        
        for it in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            # Random perturbation
            rng = np.random.default_rng()
            idx = rng.integers(0, N)
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            step_size = T * 0.01
            new_X = current_X.copy()
            new_X[idx] += step_size * direction
            new_X[idx] /= np.linalg.norm(new_X[idx])
            
            new_E = _log_energy(new_X)
            delta_E = new_E - current_E
            
            if delta_E < 0 or rng.random() < math.exp(-delta_E / max(T, 1e-10)):
                current_X = new_X
                current_E = new_E
                if current_E < best_E:
                    best_E = current_E
                    best_X = current_X.copy()
            
            T *= cooling
    
    return best_X, best_E

def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    X = _fibonacci_start()
    
    # First local optimization
    X, E = _local_optimize(X, DEADLINE - 20)
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    if remaining < 5:
        return X
    
    # Try simulated annealing from this point
    X_sa, E_sa = _simulated_annealing(X, remaining - 10, n_restarts=3)
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    # Pick the better of the two
    if E_sa < E:
        X = X_sa
        E = E_sa
    
    if remaining < 5:
        return X
    
    # Final high-precision polish
    X, E_final = _local_optimize(X, remaining - 2)
    
    # Check validity
    X = _project_to_sphere(X)
    if not np.all(np.isfinite(X)):
        return _fibonacci_start()
    
    # Check minimum distance
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    if min_dist < 1e-9:
        return _fibonacci_start()
    
    return X
