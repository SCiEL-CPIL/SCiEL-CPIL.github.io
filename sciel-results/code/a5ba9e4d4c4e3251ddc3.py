"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _energy_and_grad(X):
    """Compute logarithmic energy and its gradient.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    X = np.asarray(X)
    n = X.shape[0]
    # Compute pairwise distances
    # diff[i,j] = X[i] - X[j]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, 3)
    dists = np.linalg.norm(diff, axis=2)  # (n, n)
    # Avoid division by zero (shouldn't happen with valid configs)
    np.fill_diagonal(dists, 1.0)
    
    # Energy: -sum_{i<j} log(dists[i,j])
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient: for each i, grad_i = -sum_{j!=i} (X[i]-X[j]) / |X[i]-X[j]|^2
    # = -sum_{j!=i} diff[i,j] / dists[i,j]^2
    # diff[i,j] / dists[i,j]^2
    inv_dists_sq = 1.0 / (dists ** 2)
    np.fill_diagonal(inv_dists_sq, 0.0)
    # grad[i] = -sum_j diff[i,j] * inv_dists_sq[i,j]
    grad = -np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _project_to_sphere(X):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _energy_fsum(X):
    """Compute energy using math.fsum for accurate summation."""
    X = np.asarray(X)
    n = X.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X[i] - X[j])
            if d < 1e-15:
                return -1e18
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_X = None
    best_energy = np.inf
    
    # Multiple restarts with different initializations
    rng = np.random.RandomState(42)
    
    # Initialization 1: Fibonacci spiral
    def fib_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initialization 2: Random points
    def random_points(seed=0):
        r = np.random.RandomState(seed)
        pts = r.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts
    
    # Initialization 3: Fibonacci with perturbation
    def fib_perturbed(seed=0, sigma=0.1):
        r = np.random.RandomState(seed)
        base = fib_spiral()
        noise = r.randn(N, 3) * sigma
        X = base + noise
        return _project_to_sphere(X)
    
    initializations = [fib_spiral(), random_points(0), random_points(1), 
                       fib_perturbed(0, 0.05), fib_perturbed(1, 0.1),
                       fib_perturbed(2, 0.15), random_points(2), random_points(3)]
    
    for init_idx, X0 in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
        
        # Evaluate initial energy
        e0, g0 = _energy_and_grad(X0)
        
        # L-BFGS-B optimization
        try:
            result = minimize(
                fun=lambda x: _energy_and_grad(x)[0],
                x0=X0.flatten(),
                jac=lambda x: _energy_and_grad(x)[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 1000}
            )
            
            if result.success or result.fun < best_energy:
                X_opt = _project_to_sphere(result.x.reshape(N, 3))
                e_opt, _ = _energy_and_grad(X_opt)
                
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_X = X_opt.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 15.0:
            break
    
    # If we still have time, do a final polish on the best configuration
    if best_X is not None and time.time() < deadline - 5.0:
        try:
            result = minimize(
                fun=lambda x: _energy_and_grad(x)[0],
                x0=best_X.flatten(),
                jac=lambda x: _energy_and_grad(x)[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxfun': 2000}
            )
            X_polished = _project_to_sphere(result.x.reshape(N, 3))
            e_polished, _ = _energy_and_grad(X_polished)
            
            if e_polished < best_energy:
                best_energy = e_polished
                best_X = X_polished.copy()
        except Exception:
            pass
    
    # Final validation and cleanup
    if best_X is None:
        # Fallback to Fibonacci spiral
        best_X = fib_spiral()
    
    # Ensure points are on unit sphere and non-degenerate
    best_X = _project_to_sphere(best_X)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_X[i] - best_X[j])
            if d < 1e-9:
                # Nudge apart
                best_X[j] += 1e-8 * np.array([1, 0, 0])
                best_X[j] = best_X[j] / np.linalg.norm(best_X[j])
    
    return best_X
# EVOLVE-BLOCK-END
