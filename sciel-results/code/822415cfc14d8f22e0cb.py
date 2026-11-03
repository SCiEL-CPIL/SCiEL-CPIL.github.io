import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100
_DEADLINE = 145.0  # Safety margin before 150s hard kill
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _project(X):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return X / norms


def _energy_and_grad(X):
    """
    Calculate logarithmic energy and gradient.
    X: (N, 3) array of points on unit sphere.
    Returns: energy, gradient (N, 3)
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    """
    # Pairwise differences
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Gradient: grad_i = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    # Note: diff[i,j] = x_i - x_j, so x_j - x_i = -diff[i,j]
    inv_dist_sq = np.zeros_like(dist_sq)
    np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
    
    # grad[i] = sum_j (-diff[i,j] * inv_dist_sq[i,j])
    grad = np.sum(-diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Energy: E = -sum_{i<j} log(dist_ij)
    # = -0.5 * sum_{i!=j} log(dist_ij)
    # log(dist_ij) = 0.5 * log(dist_sq_ij)
    # So E = -0.5 * sum_{i!=j} 0.5 * log(dist_sq_ij) = -0.25 * sum_{i!=j} log(dist_sq_ij)
    # Wait, let me recompute.
    # E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # sum_{i<j} log(d_ij) = 0.5 * sum_{i!=j} log(d_ij)
    # log(d_ij) = 0.5 * log(d_ij^2) = 0.5 * log(dist_sq_ij)
    # So sum_{i<j} log(d_ij) = 0.5 * sum_{i!=j} 0.5 * log(dist_sq_ij) = 0.25 * sum_{i!=j} log(dist_sq_ij)
    # E = -0.25 * sum_{i!=j} log(dist_sq_ij)
    
    # Actually, let's be more careful.
    # E = -sum_{i<j} log(d_ij)
    # Let S = sum_{i!=j} log(d_ij) = 2 * sum_{i<j} log(d_ij)
    # So E = -S/2
    # log(d_ij) = 0.5 * log(dist_sq_ij)
    # S = sum_{i!=j} 0.5 * log(dist_sq_ij) = 0.5 * sum_{i!=j} log(dist_sq_ij)
    # E = -0.5 * S = -0.5 * 0.5 * sum_{i!=j} log(dist_sq_ij) = -0.25 * sum_{i!=j} log(dist_sq_ij)
    
    log_dist_sq = np.zeros_like(dist_sq)
    np.log(dist_sq, out=log_dist_sq, where=mask)
    energy = -0.25 * np.sum(log_dist_sq)
    
    return energy, grad


def _tangent_project(grad, X):
    """Project gradient onto tangent space of sphere at each point."""
    # grad_tangent = grad - (grad . x) * x
    dot = np.sum(grad * X, axis=1, keepdims=True)
    return grad - dot * X


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere():
    """Generate random points on unit sphere."""
    v = np.random.randn(N, 3)
    return _project(v)


def _sobol_sphere():
    """Generate quasi-random points on sphere using a simple quasi-random sequence."""
    # Use a simple Halton-like sequence for quasi-randomness
    points = np.zeros((N, 3))
    for i in range(N):
        # Simple approach: use golden ratio based angles
        t = (i + 0.5) / N
        z = 1.0 - 2.0 * t
        phi = 2.0 * np.pi * (i * (1.0 + np.sqrt(5.0)) / 2.0) % (2.0 * np.pi)
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        points[i] = [r * np.cos(phi), r * np.sin(phi), z]
    return _project(points)


def _optimize(X0, max_iter=2000, tol=1e-12):
    """Optimize using L-BFGS-B with Riemannian gradient projection."""
    x0 = X0.flatten()
    
    def fun(x):
        X = _project(x.reshape(N, 3))
        E, G = _energy_and_grad(X)
        G_tan = _tangent_project(G, X)
        return E, G_tan.flatten()
    
    options = {
        'maxiter': max_iter,
        'ftol': tol,
        'gtol': 1e-10,
        'maxls': 30,
        'maxcor': 20
    }
    
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True, options=options)
        X_opt = _project(res.x.reshape(N, 3))
        E_opt = _energy_and_grad(X_opt)[0]
        return X_opt, E_opt
    except Exception:
        return X0, _energy_and_grad(X0)[0]


def _polish(X, max_iter=5000, tol=1e-14):
    """Final high-precision polish pass."""
    x0 = X.flatten()
    
    def fun(x):
        X = _project(x.reshape(N, 3))
        E, G = _energy_and_grad(X)
        G_tan = _tangent_project(G, X)
        return E, G_tan.flatten()
    
    options = {
        'maxiter': max_iter,
        'ftol': tol,
        'gtol': 1e-12,
        'maxls': 40,
        'maxcor': 30
    }
    
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True, options=options)
        X_opt = _project(res.x.reshape(N, 3))
        E_opt = _energy_and_grad(X_opt)[0]
        return X_opt, E_opt
    except Exception:
        return X, _energy_and_grad(X)[0]


def construct():
    start_time = time.time()
    
    # Initial candidates
    X_fib = _fibonacci_spiral()
    X_sob = _sobol_sphere()
    X_rand = _random_sphere()
    
    best_X = X_fib
    best_E = _energy_and_grad(best_X)[0]
    
    # Optimize from each starting point
    starts = [X_fib, X_sob, X_rand]
    
    for X0 in starts:
        if _remaining() < 30.0:
            break
        
        X_opt, E_opt = _optimize(X0, max_iter=2000, tol=1e-12)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
    
    # Additional random restarts if time permits
    n_restarts = 0
    for _ in range(15):
        if _remaining() < 40.0:
            break
        
        X0 = _random_sphere()
        X_opt, E_opt = _optimize(X0, max_iter=1500, tol=1e-12)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
        n_restarts += 1
    
    # Final polish pass
    if _remaining() > 20.0:
        best_X, best_E = _polish(best_X, max_iter=5000, tol=1e-14)
    
    # Ensure valid output
    best_X = _project(best_X)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_X[i] - best_X[j])
            if d < 1e-9:
                # Perturb if needed
                best_X[j] = _project(best_X[j] + np.random.randn(3) * 1e-8)
    
    return best_X
