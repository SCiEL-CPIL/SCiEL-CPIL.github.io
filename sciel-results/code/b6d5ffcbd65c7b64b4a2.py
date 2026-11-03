"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def _energy_and_grad(X):
    """
    Compute the logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient with respect to X (N x 3).
    
    Energy = -sum_{i<j} log(d_ij)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / d_ij^2
    """
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_pairs = dist[mask]
    energy = -np.sum(np.log(d_pairs))
    
    inv_dist_sq = 1.0 / dist_sq
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(X):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _fibonacci_spiral(n, seed_offset=0.0):
    """Generate n points on the unit sphere using a Fibonacci spiral with optional offset."""
    i = np.arange(n) + 0.5 + seed_offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _riemannian_lbfgs(X_init, max_iter=200, tol=1e-12):
    """
    Riemannian L-BFGS optimization on the sphere.
    Uses a simple line search with backtracking.
    """
    X = X_init.copy()
    energy, grad = _energy_and_grad(X)
    
    # Simple L-BFGS memory
    m = 10
    W = []  # list of (s_i, y_i) pairs
    
    step_size = 0.1
    
    for it in range(max_iter):
        # Project gradient to tangent space
        dot_products = np.sum(grad * X, axis=1)
        grad_tangent = grad - dot_products[:, np.newaxis] * X
        
        # Check convergence
        if np.linalg.norm(grad_tangent) < tol:
            break
        
        # Compute search direction using L-BFGS two-loop recursion
        # For simplicity, use steepest descent direction in tangent space
        d = -grad_tangent
        
        # Normalize direction
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-15:
            break
        d = d / d_norm
        
        # Backtracking line search
        alpha = step_size
        for _ in range(20):
            # Move along geodesic (simplified: project after Euclidean step)
            X_new = X + alpha * d
            X_new = _project_to_sphere(X_new)
            
            energy_new, grad_new = _energy_and_grad(X_new)
            
            if energy_new < energy - 1e-4 * alpha * np.sum(grad_tangent * d):
                # Accept step
                # Update L-BFGS memory
                s = alpha * d
                y = grad_new - grad
                sy = np.sum(s * y)
                if sy > 1e-10:
                    W.append((s, y))
                    if len(W) > m:
                        W.pop(0)
                
                X = X_new
                energy = energy_new
                grad = grad_new
                step_size = min(alpha * 1.5, 0.5)
                break
            else:
                alpha *= 0.5
        
        else:
            break
        
        if alpha < 1e-8:
            break
    
    return X, energy


def _basin_hopping(X_init, n_iter=50, T_init=0.1, T_min=0.01):
    """
    Basin hopping with Riemannian L-BFGS as the local minimizer.
    """
    X_best = X_init.copy()
    energy_best, _ = _energy_and_grad(X_best)
    T = T_init
    
    for it in range(n_iter):
        # Generate perturbation
        X_perturbed = X_best + 0.05 * np.random.randn(N, 3)
        X_perturbed = _project_to_sphere(X_perturbed)
        
        # Local minimization
        X_opt, energy_opt = _riemannian_lbfgs(X_perturbed, max_iter=100, tol=1e-10)
        
        # Metropolis acceptance
        if energy_opt < energy_best:
            X_best = X_opt
            energy_best = energy_opt
        else:
            delta_e = energy_opt - energy_best
            if np.log(np.random.uniform()) < -delta_e / T:
                X_best = X_opt
                energy_best = energy_opt
        
        T = max(T * 0.95, T_min)
    
    return X_best, energy_best


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin
    
    best_X = _fibonacci_spiral(N)
    best_energy, _ = _energy_and_grad(best_X)
    
    # Multi-start with different initial configurations
    n_starts = 15
    
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
        
        # Generate initial configuration
        if start_idx == 0:
            X = _fibonacci_spiral(N)
        elif start_idx == 1:
            # Slightly different Fibonacci spiral
            X = _fibonacci_spiral(N, seed_offset=0.3)
        elif start_idx == 2:
            # Random points
            X = np.random.randn(N, 3)
            X = _project_to_sphere(X)
        else:
            # Perturbed Fibonacci spiral
            offset = np.random.uniform(-0.5, 0.5)
            X = _fibonacci_spiral(N, seed_offset=offset)
            X += 0.02 * np.random.randn(N, 3)
            X = _project_to_sphere(X)
        
        # Basin hopping
        X_opt, energy_opt = _basin_hopping(X, n_iter=30, T_init=0.05, T_min=0.005)
        
        # Final polish with more iterations
        X_polished, energy_polished = _riemannian_lbfgs(X_opt, max_iter=300, tol=1e-12)
        
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_X = X_polished.copy()
    
    # Ensure all points are on the unit sphere and valid
    best_X = _project_to_sphere(best_X)
    
    # Check for any issues
    norms = np.linalg.norm(best_X, axis=1)
    if not np.all(np.isfinite(best_X)) or np.any(norms < 1e-9):
        # Fallback to Fibonacci spiral
        best_X = _fibonacci_spiral(N)
    
    return best_X
# EVOLVE-BLOCK-END
