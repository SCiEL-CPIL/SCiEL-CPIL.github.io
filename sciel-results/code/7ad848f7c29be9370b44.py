"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

DEADLINE = 150.0  # wall-clock limit
SAFETY = 10.0     # safety margin (seconds)


def _energy_and_grad(X):
    """Compute logarithmic energy and gradient for points on the unit sphere.
    
    Energy = sum_{i<j} log(1 / ||x_i - x_j||) = - sum_{i<j} log(||x_i - x_j||)
    Gradient is computed with respect to the unconstrained coordinates,
    then projected onto the tangent space of the sphere.
    """
    n = X.shape[0]
    # Compute pairwise distances
    # diff[i,j] = X[i] - X[j]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    
    # Avoid log(0) by masking diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of -log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: d/dx_i [ -log(||x_i - x_j||) ] = -(x_i - x_j) / ||x_i - x_j||^2
    # For each pair (i,j), contribution to grad[i] is -(x_i - x_j)/dist^2
    # and to grad[j] is (x_i - x_j)/dist^2
    grad = np.zeros_like(X)
    dist_safe = dist.copy()
    np.fill_diagonal(dist_safe, 1.0)
    inv_dist2 = 1.0 / (dist_safe * dist_safe)
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j -(x_i - x_j) / dist_ij^2 = -sum_j (x_i - x_j) * inv_dist2[i,j]
    # = -x_i * sum_j inv_dist2[i,j] + sum_j x_j * inv_dist2[i,j]
    # Using matrix operations:
    # Let A = inv_dist2 (n x n), then
    # grad[i] = -A[i,:] @ X + X[i] * A[i,:].sum()
    # Actually: grad[i] = sum_j (x_j - x_i) / dist_ij^2
    # = (A @ X)[i] - X[i] * A[i,:].sum()
    
    A = inv_dist2  # (n, n)
    row_sums = A.sum(axis=1)  # (n,)
    # A @ X gives for each i: sum_j A[i,j] * X[j]
    AX = A @ X  # (n, 3)
    grad = AX - X * row_sums[:, np.newaxis]
    
    # Project gradient onto tangent space: grad_tan = grad - (grad . x) * x
    dot = np.sum(grad * X, axis=1, keepdims=True)  # (n, 1)
    grad_tan = grad - dot * X
    
    return energy, grad_tan


def _sphere_fun(X_flat):
    """Objective function for scipy minimize: flatten X, compute energy."""
    X = X_flat.reshape(N, 3)
    # Project to sphere
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    X = X / norms
    energy, _ = _energy_and_grad(X)
    return energy


def _sphere_fun_and_grad(X_flat):
    """Objective and gradient for L-BFGS-B."""
    X = X_flat.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    X = X / norms
    energy, grad_tan = _energy_and_grad(X)
    # Unproject gradient: the gradient in flat space is the tangent gradient
    # (since we're on the sphere, the normal component is zero after projection)
    grad_flat = grad_tan.reshape(-1)
    return energy, grad_flat


def _fibonacci_sphere(n, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def _lbfgs_polish(X, maxiter=2000, gtol=1e-12):
    """Polish using L-BFGS-B on the sphere via flat coordinates."""
    x0 = X.reshape(-1)
    result = minimize(
        _sphere_fun_and_grad,
        x0,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50}
    )
    X_opt = result.x.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    X_opt = X_opt / norms
    return X_opt


def _riemannian_gd(X, lr=0.01, maxiter=5000, tol=1e-10):
    """Riemannian gradient descent on the sphere with adaptive step size."""
    X = X.copy()
    best_energy, best_X = _energy_and_grad(X)[0], X.copy()
    
    for it in range(maxiter):
        energy, grad = _energy_and_grad(X)
        
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
        
        # Adaptive step size: use a simple backtracking
        step = lr
        for _ in range(20):
            X_new = X - step * grad
            # Project to sphere
            norms = np.linalg.norm(X_new, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            X_new = X_new / norms
            
            new_energy, _ = _energy_and_grad(X_new)
            if new_energy < energy:
                break
            step *= 0.5
        
        X = X_new
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
    
    return best_X, best_energy


def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    X = _fibonacci_sphere(N)
    best_X = X.copy()
    best_energy = _energy_and_grad(X)[0]
    
    # First pass: L-BFGS polish
    X = _lbfgs_polish(X, maxiter=1000, gtol=1e-10)
    energy = _energy_and_grad(X)[0]
    if energy < best_energy:
        best_energy = energy
        best_X = X.copy()
    
    # Riemannian GD with tighter tolerance
    remaining = DEADLINE - SAFETY - (time.time() - start_time)
    if remaining > 5:
        X_gd, energy_gd = _riemannian_gd(X, lr=0.05, maxiter=3000, tol=1e-11)
        if energy_gd < best_energy:
            best_energy = energy_gd
            best_X = X_gd.copy()
    
    # Final L-BFGS polish on best
    remaining = DEADLINE - SAFETY - (time.time() - start_time)
    if remaining > 5:
        X_polish = _lbfgs_polish(best_X, maxiter=2000, gtol=1e-12)
        energy_polish = _energy_and_grad(X_polish)[0]
        if energy_polish < best_energy:
            best_energy = energy_polish
            best_X = X_polish.copy()
    
    # Try a few random restarts if time permits
    remaining = DEADLINE - SAFETY - (time.time() - start_time)
    for seed in [42, 123, 456]:
        if remaining < 10:
            break
        X_rand = _random_sphere(N, seed)
        X_p = _lbfgs_polish(X_rand, maxiter=500, gtol=1e-8)
        energy_p = _energy_and_grad(X_p)[0]
        if energy_p < best_energy:
            best_energy = energy_p
            best_X = X_p.copy()
            # Polish the improved solution
            remaining = DEADLINE - SAFETY - (time.time() - start_time)
            if remaining > 5:
                X_pp = _lbfgs_polish(best_X, maxiter=1000, gtol=1e-10)
                energy_pp = _energy_and_grad(X_pp)[0]
                if energy_pp < best_energy:
                    best_energy = energy_pp
                    best_X = X_pp.copy()
        remaining = DEADLINE - SAFETY - (time.time() - start_time)
    
    # Ensure all points are on the unit sphere and valid
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_X = best_X / norms
    
    return best_X
# EVOLVE-BLOCK-END
