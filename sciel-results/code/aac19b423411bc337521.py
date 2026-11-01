"""Optimize logarithmic energy of 100 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def _energy_and_grad(X):
    """Compute energy and gradient for points on sphere.
    X: (N, 3) array of unit vectors.
    Returns (energy, grad) where grad is (N, 3) tangent-space gradient.
    """
    n = X.shape[0]
    # Compute pairwise distances
    # X[:, None, :] - X[None, :, :] has shape (n, n, 3)
    diff = X[:, None, :] - X[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid division by zero for i==j
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: d/dx_i [-log(|x_i - x_j|)] = (x_i - x_j) / |x_i - x_j|^2
    # For each i, sum over j != i of (x_i - x_j) / dist^2
    # grad[i] = sum_{j != i} (x_i - x_j) / dist_sq[i,j]
    # Note: dist_sq[i,j] = dist_sq[j,i]
    
    # grad[i] = sum_j (x_i - x_j) / dist_sq[i,j] for j != i
    # = x_i * sum_j 1/dist_sq[i,j] - sum_j x_j / dist_sq[i,j]
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # sum_j 1/dist_sq[i,j] for each i
    row_sums = np.sum(inv_dist_sq, axis=1)  # shape (n,)
    
    # sum_j x_j / dist_sq[i,j] for each i
    # This is X @ inv_dist_sq.T, but inv_dist_sq is symmetric
    cross_term = inv_dist_sq @ X  # shape (n, 3)
    
    grad = row_sums[:, None] * X - cross_term
    
    # Project gradient to tangent space: grad_tangent = grad - (grad . x_i) * x_i
    grad_dot_x = np.sum(grad * X, axis=1)  # shape (n,)
    grad_tangent = grad - grad_dot_x[:, None] * X
    
    return energy, grad_tangent

def _energy_fsum(X):
    """Compute energy using math.fsum for accuracy."""
    n = X.shape[0]
    total = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X[i] - X[j])
            if d < 1e-15:
                return float('inf')
            total.append(-math.log(d))
    return math.fsum(total)

def _project_to_sphere(X):
    """Normalize rows of X to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _riemannian_gd(X, max_iters=2000, tol=1e-12, time_limit=20.0):
    """Riemannian gradient descent with backtracking line search."""
    start_time = time.time()
    X = _project_to_sphere(X.copy())
    energy, grad = _energy_and_grad(X)
    best_X = X.copy()
    best_energy = energy
    
    step_size = 0.1
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - 10:
            break
        
        # Backtracking line search
        alpha = step_size
        for _ in range(30):
            # Update: X_new = X + alpha * grad, then project
            X_new = X + alpha * grad
            X_new = _project_to_sphere(X_new)
            
            new_energy, new_grad = _energy_and_grad(X_new)
            
            if new_energy < energy - 1e-4 * alpha * np.sum(grad * grad):
                break
            alpha *= 0.5
        
        X = X_new
        energy = new_energy
        grad = new_grad
        
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
        
        # Decay step size slowly
        step_size *= 0.999
    
    return _project_to_sphere(best_X), best_energy

def _l_bfgs_polish(X, time_limit=15.0):
    """Polish with L-BFGS-B in ambient space, projecting back to sphere."""
    start_time = time.time()
    
    def fun_and_grad_flat(flat):
        if time.time() - start_time > time_limit:
            return float('inf'), np.zeros_like(flat)
        X = flat.reshape(N, 3)
        X = _project_to_sphere(X)
        energy, grad = _energy_and_grad(X)
        return energy, grad.flatten()
    
    x0 = X.flatten()
    result = minimize(
        fun_and_grad_flat,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
    )
    
    X_opt = _project_to_sphere(result.x.reshape(N, 3))
    energy = _energy_fsum(X_opt)
    
    return X_opt, energy

def construct():
    start_time = time.time()
    
    # Multiple initializations
    best_X = None
    best_energy = float('inf')
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    X_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    candidates = [X_fib]
    
    # 2. Random initializations with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    for _ in range(3):
        X_rand = rng.randn(N, 3)
        X_rand = X_rand / np.linalg.norm(X_rand, axis=1, keepdims=True)
        candidates.append(X_rand)
    
    # 3. Perturbed Fibonacci
    X_pert = X_fib + 0.1 * rng.randn(N, 3)
    X_pert = _project_to_sphere(X_pert)
    candidates.append(X_pert)
    
    for X0 in candidates:
        if time.time() > DEADLINE - 20:
            break
        
        # Riemannian GD
        X_gd, e_gd = _riemannian_gd(X0, max_iters=3000, tol=1e-14, time_limit=15.0)
        
        if time.time() > DEADLINE - 15:
            # Just use GD result
            if e_gd < best_energy:
                best_energy = e_gd
                best_X = X_gd
            continue
        
        # L-BFGS polish
        X_pol, e_pol = _l_bfgs_polish(X_gd, time_limit=10.0)
        
        if e_pol < best_energy:
            best_energy = e_pol
            best_X = X_pol
    
    # Final validation and return
    if best_X is None:
        best_X = X_fib
    
    # Ensure all points are valid
    best_X = _project_to_sphere(best_X)
    norms = np.linalg.norm(best_X, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_X)):
        best_X = X_fib
    
    return best_X
