"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|xi-xj|) using math.fsum."""
    n = len(points)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_grad(points):
    """Compute gradient of logarithmic energy w.r.t. points (N,3)."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                continue
            coeff = -1.0 / (d * d)
            grad[i] += coeff * diff
            grad[j] -= coeff * diff
    return grad


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _vectorized_energy(X):
    """Vectorized energy computation."""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    return -np.sum(np.log(dist[mask]))


def _vectorized_gradient(X):
    """Vectorized gradient computation."""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-12)
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    return grad


def _cartesian_to_spherical(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    theta = np.arctan2(y, x)
    phi = np.arccos(np.clip(z / np.linalg.norm(X, axis=1), -1.0, 1.0))
    return np.column_stack([theta, phi])


def _spherical_to_cartesian(params):
    theta, phi = params[:, 0], params[:, 1]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def _energy_grad_spherical(params):
    X = _spherical_to_cartesian(params)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    e = -np.sum(np.log(dist[mask]))
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad_cart = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    theta, phi = params[:, 0], params[:, 1]
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    grad_theta = -sin_phi * sin_theta * grad_cart[:, 0] + sin_phi * cos_theta * grad_cart[:, 1]
    grad_phi = cos_phi * cos_theta * grad_cart[:, 0] + cos_phi * sin_theta * grad_cart[:, 1] - sin_phi * grad_cart[:, 2]
    
    mask_pole = sin_phi < 1e-6
    grad_theta[mask_pole] = 0.0
    
    grad_params = np.column_stack([grad_theta, grad_phi])
    return e, grad_params.flatten()


def _optimize_spherical(X_init, time_limit):
    params_init = _cartesian_to_spherical(X_init)
    x0 = params_init.flatten()
    
    def fun(x):
        e, _ = _energy_grad_spherical(x.reshape(N, 2))
        return e
    
    def jac(x):
        _, g = _energy_grad_spherical(x.reshape(N, 2))
        return g
    
    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
        best_params = res.x.reshape(N, 2)
        best_X = _spherical_to_cartesian(best_params)
        best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
        return best_X, _vectorized_energy(best_X)
    except Exception:
        return X_init, _vectorized_energy(X_init)


def _gradient_descent_polish(points, time_limit, lr_init=1e-4, min_lr=1e-8):
    """Gradient descent polish with adaptive step size."""
    lr = lr_init
    best_points = points.copy()
    best_energy = _vectorized_energy(points)
    patience = 0
    max_patience = 100
    
    while time.time() < time_limit:
        grad = _vectorized_gradient(points)
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        new_points = _project_to_sphere(points - lr * grad_tangent)
        new_energy = _vectorized_energy(new_points)
        
        if new_energy < best_energy:
            best_points = new_points.copy()
            best_energy = new_energy
            patience = 0
            lr = min(lr * 1.02, 0.05)
        else:
            patience += 1
            lr = max(lr * 0.95, min_lr)
        
        points = new_points
        
        if patience > max_patience:
            break
    
    return best_points, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_overall_X = X_fib.copy()
    best_overall_energy = _vectorized_energy(X_fib)
    
    # Phase 1: L-BFGS optimization from Fibonacci spiral
    time_per_restart = (deadline - time.time()) / 6.0
    X_best, e_best = _optimize_spherical(X_fib, time.time() + time_per_restart)
    if e_best < best_overall_energy:
        best_overall_energy = e_best
        best_overall_X = X_best.copy()
    
    # Phase 2: Random restarts with L-BFGS
    rng = np.random.default_rng(42)
    for k in range(4):
        if time.time() > deadline:
            break
        
        X_rand = rng.normal(size=(N, 3))
        X_rand = X_rand / np.linalg.norm(X_rand, axis=1, keepdims=True)
        
        time_per_restart = (deadline - time.time()) / (6 - k - 1)
        X_best, e_best = _optimize_spherical(X_rand, time.time() + time_per_restart)
        if e_best < best_overall_energy:
            best_overall_energy = e_best
            best_overall_X = X_best.copy()
    
    # Phase 3: Gradient descent polish on best solution
    if time.time() < deadline - 5.0:
        polish_deadline = deadline - 2.0
        X_polished, e_polished = _gradient_descent_polish(best_overall_X, polish_deadline)
        if e_polished < best_overall_energy:
            best_overall_energy = e_polished
            best_overall_X = X_polished.copy()
    
    # Final verification with exact energy
    final_energy = _log_energy(best_overall_X)
    if math.isinf(final_energy):
        # Fallback to Fibonacci if invalid
        return X_fib
    
    return best_overall_X
# EVOLVE-BLOCK-END
