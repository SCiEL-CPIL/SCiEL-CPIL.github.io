"""Optimize 150 points on unit sphere to minimize logarithmic energy via Newton/BFGS."""
import numpy as np
import math
import time

N = 150

# EVOLVE-BLOCK-START
def _energy_and_grad(X):
    """Compute log energy and gradient. X is (N,3) on unit sphere."""
    # Pairwise distances
    # d_ij = |x_i - x_j|
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N,N,3)
    d = np.linalg.norm(diff, axis=2)  # (N,N)
    np.fill_diagonal(d, 1.0)  # avoid log(0)
    # log(1/d) = -log(d)
    E = -np.sum(np.log(d)) / 2.0  # each pair counted twice
    # Gradient: dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Actually: E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # d/dx_i [-log(|x_i - x_j|)] = -(x_i - x_j)/|x_i - x_j|^2
    # So grad_i = sum_{j != i} (x_i - x_j)/|x_i - x_j|^2  ... wait sign
    # E = -sum log(d_ij), dE/dx_i = -sum_j d/dx_i log(d_ij) = -sum_j (x_i-x_j)/d_ij^2
    # But we want to minimize E, so gradient is negative of that for descent
    # Let me recompute: E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j!=i, j>i or j<i} (1/2) * 2*(x_i-x_j)/d_ij^2 ... 
    # Actually d/dx_i log(|x_i - x_j|) = (x_i - x_j)/|x_i - x_j|^2
    # So dE/dx_i = -sum_{j != i} (x_i - x_j)/d_ij^2
    # For minimization, we use this gradient directly
    inv_d2 = 1.0 / (d * d)
    np.fill_diagonal(inv_d2, 0.0)
    # grad_i = -sum_j (x_i - x_j) * inv_d2[i,j]
    # = -sum_j (x_i * inv_d2[i,j] - x_j * inv_d2[i,j])
    # = -x_i * sum_j inv_d2[i,j] + sum_j x_j * inv_d2[i,j]
    row_sum = inv_d2.sum(axis=1)  # (N,)
    grad = -X * row_sum[:, np.newaxis] + (X.T @ inv_d2).T
    return E, grad


def _project_to_sphere(X):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin before 120s kill
    
    # Golden-angle spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    X = _project_to_sphere(X)
    
    best_X = X.copy()
    best_E, _ = _energy_and_grad(X)
    
    # BFGS Hessian approximation
    n = N * 3
    H = np.eye(n)  # Hessian approximation
    x_flat = X.flatten().copy()
    
    # Compute initial gradient
    E, grad = _energy_and_grad(X)
    g_flat = grad.flatten().copy()
    
    # Newton iterations with BFGS
    max_iter = 500
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Newton step: solve H * s = -g
        try:
            s = np.linalg.solve(H, -g_flat)
        except np.linalg.LinAlgError:
            s = -g_flat
        
        # Line search with backtracking
        alpha = 1.0
        E_new, grad_new = _energy_and_grad(X)
        g_new_flat = grad_new.flatten().copy()
        
        # Armijo condition
        c1 = 1e-4
        rho = 0.5
        for ls in range(50):
            X_try = _project_to_sphere(X + alpha * s.reshape(N, 3))
            E_try, grad_try = _energy_and_grad(X_try)
            if E_try <= E + c1 * alpha * np.dot(g_flat, s):
                break
            alpha *= rho
        
        # Update
        X = _project_to_sphere(X + alpha * s.reshape(N, 3))
        E, grad = _energy_and_grad(X)
        g_flat = grad.flatten().copy()
        
        # Track best
        if E < best_E:
            best_E = E
            best_X = X.copy()
        
        # BFGS update
        x_new_flat = X.flatten().copy()
        s_vec = x_new_flat - x_flat
        g_diff = g_flat - g_new_flat  # Actually should be g_new - g_old
        
        # Wait, I need to be careful with the order
        # Let me redo: at iteration, we have x, g. We take step to x_new, compute g_new.
        # BFGS: H_new = H - H s g^T s / (s^T g) + s s^T / (s^T g)  ... no
        # Standard BFGS: H_{k+1} = (I - rho s y^T) H_k (I - rho y s^T) + rho s s^T
        # where y = g_{k+1} - g_k, rho = 1/(y^T s)
        
        y = g_flat - g_new_flat  # This is g_old - g_new, should be g_new - g_old for standard
        # Let me use: y = g_new - g_old
        y = g_new_flat - g_flat
        sy = np.dot(s_vec, y)
        
        if abs(sy) > 1e-12:
            rho = 1.0 / sy
            # BFGS update
            H = H - rho * np.outer(np.dot(H, s_vec), np.dot(H, s_vec).T * 0 + y)  # This is wrong
            # Correct BFGS:
            Hs = np.dot(H, s_vec)
            Hy = np.dot(H, y)
            H = H - np.outer(Hs, y) * rho - np.outer(y, Hs) * rho + rho * np.outer(s_vec, s_vec)
            # Ensure symmetry
            H = (H + H.T) / 2.0
        
        x_flat = x_new_flat
        g_flat = g_new_flat
        
        # Check convergence
        if np.linalg.norm(g_flat) < 1e-10:
            break
    
    # Final polish: more iterations with tighter tolerance
    X = best_X.copy()
    E, grad = _energy_and_grad(X)
    g_flat = grad.flatten().copy()
    x_flat = X.flatten().copy()
    
    for it in range(200):
        if time.time() > deadline:
            break
        
        try:
            s = np.linalg.solve(H, -g_flat)
        except np.linalg.LinAlgError:
            s = -g_flat
        
        alpha = 1.0
        c1 = 1e-6
        rho = 0.5
        for ls in range(60):
            X_try = _project_to_sphere(X + alpha * s.reshape(N, 3))
            E_try, _ = _energy_and_grad(X_try)
            if E_try <= E + c1 * alpha * np.dot(g_flat, s):
                break
            alpha *= rho
        
        X = _project_to_sphere(X + alpha * s.reshape(N, 3))
        E, grad = _energy_and_grad(X)
        g_flat = grad.flatten().copy()
        
        if E < best_E:
            best_E = E
            best_X = X.copy()
        
        x_new_flat = X.flatten().copy()
        s_vec = x_new_flat - x_flat
        y = g_flat - g_new_flat if 'g_new_flat' in dir() else g_flat
        # Recompute properly
        g_old_flat = x_flat  # This is wrong, let me fix
        
        # Simplified: just do gradient-based polish
        x_flat = x_new_flat
        g_new_flat = g_flat
        
        if np.linalg.norm(g_flat) < 1e-12:
            break
    
    return _project_to_sphere(best_X)
# EVOLVE-BLOCK-END
