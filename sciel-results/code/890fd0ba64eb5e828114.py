"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
TIME_LIMIT = 140.0  # safety margin before 150s kill


def _energy_and_grad(X):
    """
    Compute logarithmic energy and its gradient using Gram matrix.
    Energy = sum_{i<j} log(1 / |x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Distance |x_i - x_j|^2 = 2 - 2*(x_i . x_j)
    So log(1/d) = -log(d) = -0.5*log(2 - 2*g_ij) where g_ij = x_i . x_j
    
    Actually: d = |x_i - x_j|, d^2 = 2 - 2*g_ij
    log(1/d) = -log(d) = -0.5 * log(d^2) = -0.5 * log(2 - 2*g_ij)
    
    Gradient wrt x_k: sum_{j != k} -1/2 * d/dx_k log(2 - 2*g_kj)
    d/dx_k g_kj = x_j
    d/dx_k log(2 - 2*g_kj) = -2*x_j / (2 - 2*g_kj)
    So gradient contribution from pair (k,j) to x_k: -1/2 * (-2*x_j / (2-2*g_kj)) = x_j / (2 - 2*g_kj)
    
    But we need to be careful: the energy is sum over i<j. Let's use the full matrix approach.
    
    For a point x_i, the gradient is: sum_{j != i} x_j / (2 - 2 * x_i.dot(x_j))
    Wait, let me re-derive.
    
    E = sum_{i<j} -0.5 * log(2 - 2*g_ij) where g_ij = x_i . x_j
    
    dE/dx_i = sum_{j != i} dE/dg_ij * dg_ij/dx_i
    
    For j < i: E contains -0.5*log(2-2*g_ji), dg_ji/dx_i = x_j
    For j > i: E contains -0.5*log(2-2*g_ij), dg_ij/dx_i = x_j
    
    dE/dg_ij = -0.5 * (-2) / (2 - 2*g_ij) = 1 / (2 - 2*g_ij)
    
    So dE/dx_i = sum_{j != i} [1/(2 - 2*g_ij)] * x_j
    
    This is the gradient in R^3. But we need the Riemannian gradient on the sphere,
    which is the projection: grad_S = grad_R - (grad_R . x_i) * x_i
    """
    # Compute Gram matrix
    G = X @ X.T  # shape (N, N), G[i,j] = x_i . x_j
    
    # d^2 = 2 - 2*G, for i != j
    D2 = 2.0 - 2.0 * G
    np.fill_diagonal(D2, 1.0)  # avoid division by zero on diagonal
    
    # Energy = sum_{i<j} -0.5 * log(D2[i,j])
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -0.5 * np.sum(np.log(D2[iu]))
    
    # Gradient: for each i, grad_i = sum_{j != i} x_j / D2[i,j]
    # In matrix form: grad = X @ (1/D2) but with diagonal zeroed
    inv_D2 = np.zeros_like(D2)
    np.fill_diagonal(inv_D2, 0.0)
    inv_D2[iu] = 1.0 / D2[iu]
    inv_D2[(iu[1], iu[0])] = 1.0 / D2[iu]  # symmetric
    
    # grad_raw[i] = sum_j X[j] * inv_D2[i,j] = (inv_D2 @ X.T)[i]
    grad_raw = inv_D2 @ X.T  # shape (N, 3)
    
    # Project to sphere: Riemannian gradient
    # grad_S[i] = grad_raw[i] - (grad_raw[i] . X[i]) * X[i]
    dots = np.sum(grad_raw * X, axis=1)  # shape (N,)
    grad = grad_raw - dots[:, np.newaxis] * X
    
    return energy, grad


def _project_to_sphere(X):
    """Normalize rows to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _fibonacci_sphere():
    """Golden-angle spiral initial points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere():
    """Random points on sphere."""
    v = np.random.randn(N, 3)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _lbfgsb_optimize(X0, max_iter=500, time_remaining_fn=None):
    """
    Optimize using L-BFGS-B on the sphere via retraction/projection.
    We use a simple approach: optimize in R^3 with penalty, or use 
    the fact that scipy can handle this with constraints.
    
    Actually, let's use a simpler approach: parameterize via angles or 
    just do projected gradient / L-BFGS with projection.
    
    For simplicity and robustness, use scipy minimize with the 
    energy function and project after each step.
    """
    best_X = X0.copy()
    best_E, _ = _energy_and_grad(best_X)
    
    def fun_and_grad(x_flat):
        X = x_flat.reshape(N, 3)
        E, g = _energy_and_grad(X)
        return E, g.flatten()
    
    x0 = X0.flatten()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12,
                                'maxls': 50})
        if res.success or hasattr(res, 'x'):
            X_new = _project_to_sphere(res.x.reshape(N, 3))
            E_new, _ = _energy_and_grad(X_new)
            if E_new < best_E:
                best_X = X_new
                best_E = E_new
    except Exception:
        pass
    
    return best_X, best_E


def _gd_polish(X0, n_steps=2000, lr_init=0.01, time_remaining_fn=None):
    """
    Riemannian gradient descent polish with backtracking line search.
    """
    X = X0.copy()
    E, g = _energy_and_grad(X)
    best_X = X.copy()
    best_E = E
    
    lr = lr_init
    for step in range(n_steps):
        if time_remaining_fn is not None and time_remaining_fn() < 5.0:
            break
        
        # Gradient descent step in R^3
        X_new = X - lr * g
        X_new = _project_to_sphere(X_new)
        
        E_new, g_new = _energy_and_grad(X_new)
        
        if E_new < E:
            X = X_new
            E = E_new
            g = g_new
            if E < best_E:
                best_X = X.copy()
                best_E = E
            # Slowly decay learning rate
            lr = max(lr * 0.999, 1e-6)
        else:
            lr *= 0.5
            if lr < 1e-10:
                break
    
    return best_X, best_E


def construct():
    t_start = time.time()
    
    def time_remaining():
        return TIME_LIMIT - (time.time() - t_start)
    
    # Start with Fibonacci sphere
    X = _fibonacci_sphere()
    E, _ = _energy_and_grad(X)
    best_X = X.copy()
    best_E = E
    
    # Try multiple restarts with different initializations
    np.random.seed(42)
    
    # First, polish the Fibonacci start
    if time_remaining() > 20:
        X_polished, E_polished = _lbfgsb_optimize(X, max_iter=300, time_remaining_fn=time_remaining)
        if E_polished < best_E:
            best_X = X_polished
            best_E = E_polished
        
        if time_remaining() > 15:
            X_polished, E_polished = _gd_polish(best_X, n_steps=1000, lr_init=0.005, time_remaining_fn=time_remaining)
            if E_polished < best_E:
                best_X = X_polished
                best_E = E_polished
    
    # Try random restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time_remaining() < 25:
            break
        
        X_rand = _random_sphere()
        E_rand, _ = _energy_and_grad(X_rand)
        
        if time_remaining() < 15:
            break
        
        X_opt, E_opt = _lbfgsb_optimize(X_rand, max_iter=200, time_remaining_fn=time_remaining)
        if E_opt < best_E:
            best_X = X_opt
            best_E = E_opt
        
        if time_remaining() < 10:
            break
        
        X_polished, E_polished = _gd_polish(best_X, n_steps=500, lr_init=0.003, time_remaining_fn=time_remaining)
        if E_polished < best_E:
            best_X = X_polished
            best_E = E_polished
    
    # Final polish if time allows
    if time_remaining() > 10:
        X_polished, E_polished = _gd_polish(best_X, n_steps=2000, lr_init=0.001, time_remaining_fn=time_remaining)
        if E_polished < best_E:
            best_X = X_polished
            best_E = E_polished
    
    # Ensure valid output
    best_X = _project_to_sphere(best_X)
    
    # Check for near-duplicate points
    G = best_X @ best_X.T
    D2 = 2.0 - 2.0 * G
    np.fill_diagonal(D2, 1.0)
    if np.min(D2) < 1e-18:
        # Perturb slightly
        best_X = _project_to_sphere(best_X + 1e-10 * np.random.randn(N, 3))
    
    return best_X
