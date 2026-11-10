import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

def _energy_and_grad(X):
    """
    Compute logarithmic energy and Euclidean gradient for points on unit sphere.
    X: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where grad is (N, 3).
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = X.shape[0]
    # Compute pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    dist = np.sqrt(np.maximum(dist2, 1e-30))  # (n, n)
    
    # Avoid log(0) and division by zero on diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    energy = -np.sum(np.log(dist[triu_mask]))
    
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    dist2_safe = dist2.copy()
    np.fill_diagonal(dist2_safe, 1.0)
    
    inv_dist2 = np.zeros_like(dist2_safe)
    inv_dist2[mask] = 1.0 / dist2_safe[mask]
    
    sum_inv_dist2 = np.sum(inv_dist2, axis=1)  # (n,)
    sum_X_over_dist2 = inv_dist2 @ X  # (n, 3)
    
    grad = X * sum_inv_dist2[:, np.newaxis] - sum_X_over_dist2  # (n, 3)
    
    return energy, grad

def _spherical_to_cartesian(params):
    """
    Convert spherical coordinates (theta, phi) to Cartesian.
    params: (2N,) array [theta_0, phi_0, theta_1, phi_1, ...]
    Returns: (N, 3) array
    """
    n = params.shape[0] // 2
    theta = params[0::2]
    phi = params[1::2]
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return np.column_stack([x, y, z])

def _cartesian_to_spherical(X):
    """
    Convert Cartesian to spherical coordinates.
    X: (N, 3)
    Returns: (2N,) array
    """
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    # Ensure phi is in [0, 2*pi)
    phi = phi % (2 * np.pi)
    
    params = np.zeros(2 * X.shape[0])
    params[0::2] = theta
    params[1::2] = phi
    return params

def _objective_spherical(params):
    """
    Objective function for spherical coordinates.
    """
    X = _spherical_to_cartesian(params)
    energy, _ = _energy_and_grad(X)
    return energy

def _gradient_spherical(params):
    """
    Gradient of objective w.r.t. spherical coordinates.
    """
    n = params.shape[0] // 2
    theta = params[0::2]
    phi = params[1::2]
    
    X = _spherical_to_cartesian(params)
    energy, grad_euclidean = _energy_and_grad(X)
    
    # Project Euclidean gradient to tangent space
    # g_T = g - (g . x) x
    dot = np.sum(grad_euclidean * X, axis=1)
    grad_tangent = grad_euclidean - dot[:, np.newaxis] * X
    
    # Basis vectors
    # dX/dtheta = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
    # dX/dphi = (-sin(theta)sin(phi), sin(theta)cos(phi), 0)
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    
    dX_dtheta = np.column_stack([cos_t * cos_p, cos_t * sin_p, -sin_t])
    dX_dphi = np.column_stack([-sin_t * sin_p, sin_t * cos_p, np.zeros(n)])
    
    # Gradient components
    grad_theta = np.sum(grad_tangent * dX_dtheta, axis=1)
    grad_phi = np.sum(grad_tangent * dX_dphi, axis=1)
    
    grad_params = np.zeros(2 * n)
    grad_params[0::2] = grad_theta
    grad_params[1::2] = grad_phi
    
    return grad_params

def _fibonacci_spiral(n, offset=0.5):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_X = _fibonacci_spiral(N)
    best_energy, _ = _energy_and_grad(best_X)
    
    rng = np.random.default_rng(42)
    
    def _optimize(params_init, time_budget):
        nonlocal best_X, best_energy
        nonlocal start_time, deadline
        
        def callback(params):
            if time.time() > time_budget:
                raise RuntimeError("time_limit")
            return None
        
        try:
            res = minimize(
                _objective_spherical,
                params_init,
                jac=_gradient_spherical,
                method='L-BFGS-B',
                bounds=[(0, np.pi)] * N + [(0, 2*np.pi)] * N,
                options={
                    'maxiter': 200,
                    'maxls': 20,
                    'gtol': 1e-8,
                    'ftol': 1e-12,
                    'maxfun': 500,
                },
                callback=callback,
            )
            X_opt = _spherical_to_cartesian(res.x)
            # Normalize just in case
            norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
            X_opt = X_opt / norms
            
            e_opt, _ = _energy_and_grad(X_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_X = X_opt
        except RuntimeError:
            pass
        except Exception:
            pass
        
        return best_X, best_energy
    
    # 1. Optimize from Fibonacci spiral
    params_fib = _cartesian_to_spherical(_fibonacci_spiral(N))
    _optimize(params_fib, deadline - 30)
    
    # 2. Try random restarts
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Generate random spherical coordinates
        theta_rand = rng.uniform(0.1, np.pi - 0.1, N)
        phi_rand = rng.uniform(0, 2 * np.pi, N)
        params_rand = np.zeros(2 * N)
        params_rand[0::2] = theta_rand
        params_rand[1::2] = phi_rand
        
        _optimize(params_rand, deadline - 20)
        
        # Also try perturbing the best solution
        if r % 2 == 0:
            X_perturbed = best_X + 0.5 * rng.standard_normal((N, 3))
            X_perturbed = X_perturbed / np.linalg.norm(X_perturbed, axis=1, keepdims=True)
            params_pert = _cartesian_to_spherical(X_perturbed)
            _optimize(params_pert, deadline - 20)
            
    # Final polish: run one more optimization from the best solution with tighter tolerances
    if time.time() < deadline - 5:
        params_best = _cartesian_to_spherical(best_X)
        
        def _objective_polish(params):
            X = _spherical_to_cartesian(params)
            energy, _ = _energy_and_grad(X)
            return energy
            
        def _gradient_polish(params):
            return _gradient_spherical(params)
            
        try:
            res = minimize(
                _objective_polish,
                params_best,
                jac=_gradient_polish,
                method='L-BFGS-B',
                bounds=[(0, np.pi)] * N + [(0, 2*np.pi)] * N,
                options={
                    'maxiter': 100,
                    'maxls': 20,
                    'gtol': 1e-10,
                    'ftol': 1e-14,
                    'maxfun': 200,
                }
            )
            X_opt = _spherical_to_cartesian(res.x)
            norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
            X_opt = X_opt / norms
            e_opt, _ = _energy_and_grad(X_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_X = X_opt
        except Exception:
            pass

    # Verify
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    if not np.all(np.isfinite(best_X)):
        best_X = _fibonacci_spiral(N)
        
    return best_X
