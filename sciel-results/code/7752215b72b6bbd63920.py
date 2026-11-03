import numpy as np
from scipy.optimize import minimize
import time
import math

N = 150

def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy_and_grad(X):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    dE/dxi = -sum_{j≠i} (xi-xj)/|xi-xj|^2
    """
    n = X.shape[0]
    # Compute all pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: dE/dxi = -sum_{j≠i} (xi-xj)/|xi-xj|^2
    with np.errstate(divide='ignore', invalid='ignore'):
        grad_term = diff / (dist_sq[:, :, np.newaxis] + 1e-30)
    np.fill_diagonal(grad_term[:, :, 0], 0)
    np.fill_diagonal(grad_term[:, :, 1], 0)
    np.fill_diagonal(grad_term[:, :, 2], 0)
    
    grad = -np.sum(grad_term, axis=1)  # (n, 3)
    
    return energy, grad

def _energy_only(X):
    """Compute only the logarithmic energy."""
    n = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))

def _project_to_sphere(X):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return X / norms

def _exp_map(X, V):
    """Exponential map on the unit sphere.
    
    X: current point on sphere (n, 3)
    V: tangent vector at X (n, 3), i.e., V[i] dot X[i] = 0
    Returns: new point on sphere after moving along V
    """
    # For unit sphere, exp_X(V) = X*cos(||V||) + (V/||V||)*sin(||V||)
    v_norms = np.linalg.norm(V, axis=1, keepdims=True)
    # Handle small norms to avoid division by zero
    small_mask = (v_norms < 1e-10).flatten()
    
    result = np.zeros_like(X)
    
    # For points with small V, use Taylor expansion: X + V + 0.5*(X*||V||^2)
    if np.any(small_mask):
        V_small = V[small_mask]
        X_small = X[small_mask]
        result[small_mask] = X_small + V_small + 0.5 * X_small * np.sum(V_small * V_small, axis=1, keepdims=True)
    
    # For points with larger V, use exact formula
    if np.any(~small_mask):
        V_large = V[~small_mask]
        X_large = X[~small_mask]
        v_norms_large = v_norms[~small_mask].flatten()
        cos_theta = np.cos(v_norms_large)
        sin_theta = np.sin(v_norms_large)
        result[~small_mask] = X_large * cos_theta[:, np.newaxis] + V_large * (sin_theta / v_norms_large)[:, np.newaxis]
    
    return _project_to_sphere(result)

def _riemannian_cg(X, max_iter=200, tol=1e-12, time_limit=None, start_time=None):
    """Riemannian conjugate gradient method on the unit sphere.
    
    Uses exponential map for updates and line search for step size.
    """
    if time_limit is None:
        time_limit = 100.0
    if start_time is None:
        start_time = time.time()
    
    n = X.shape[0]
    X = _project_to_sphere(X.copy())
    
    energy, grad = _energy_and_grad(X)
    
    # Project gradient to tangent space: grad_t = grad - (grad . X) * X
    grad_t = grad - np.sum(grad * X, axis=1, keepdims=True) * X
    
    # Initial search direction = -grad_t
    d = -grad_t
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-15:
        return X, energy
    
    d = d / d_norm  # normalize direction
    
    prev_grad_t = grad_t.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Check convergence
        if np.linalg.norm(grad_t) < tol:
            break
        
        # Line search: find step size alpha
        alpha = 0.1
        best_alpha = 0.0
        best_energy = energy
        best_X = X
        
        for ls in range(20):
            # Exponential map update
            X_new = _exp_map(X, alpha * d)
            e_new, g_new = _energy_and_grad(X_new)
            
            if e_new < energy - 1e-4 * alpha * np.sum(grad_t * d):
                # Accept step
                best_alpha = alpha
                best_energy = e_new
                best_X = X_new
                break
            else:
                best_alpha = alpha
                best_energy = min(best_energy, e_new)
                best_X = X_new if e_new < best_energy else best_X
                alpha *= 0.5
        
        X = best_X
        energy = best_energy
        
        # Compute new gradient and project to tangent space
        _, grad_new = _energy_and_grad(X)
        grad_new_t = grad_new - np.sum(grad_new * X, axis=1, keepdims=True) * X
        
        # Check if we made progress
        if energy >= best_energy - 1e-15:
            # Reset conjugate gradient
            d = -grad_new_t
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-15:
                break
            d = d / d_norm
            prev_grad_t = grad_new_t.copy()
            continue
        
        # Fletcher-Reeves beta
        beta = np.sum(grad_new_t * grad_new_t) / max(np.sum(prev_grad_t * prev_grad_t), 1e-30)
        
        # Ensure beta is non-negative
        beta = max(beta, 0.0)
        
        # New search direction
        d_new = -grad_new_t + beta * d
        
        # Check if direction is still descending
        if np.sum(d_new * grad_new_t) > 0:
            # Reset to steepest descent
            d_new = -grad_new_t
        
        d_norm = np.linalg.norm(d_new)
        if d_norm < 1e-15:
            break
        d = d_new / d_norm
        
        prev_grad_t = grad_new_t.copy()
    
    return X, energy

def _lbfgs_polish(X, time_limit=30.0, start_time=None):
    """Final L-BFGS polish using scipy's minimize."""
    if start_time is None:
        start_time = time.time()
    
    if time.time() - start_time > time_limit:
        return X
    
    # Flatten to 1D for scipy
    def objective_flat(x_flat):
        X_local = x_flat.reshape(N, 3)
        X_local = _project_to_sphere(X_local)
        e, _ = _energy_and_grad(X_local)
        return e
    
    def gradient_flat(x_flat):
        X_local = x_flat.reshape(N, 3)
        X_local = _project_to_sphere(X_local)
        _, g = _energy_and_grad(X_local)
        # Project to tangent space
        g_t = g - np.sum(g * X_local, axis=1, keepdims=True) * X_local
        return g_t.flatten()
    
    x0 = X.flatten()
    
    try:
        res = minimize(
            objective_flat,
            x0,
            jac=gradient_flat,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
        )
        
        if res.success or res.fun < _energy_only(X):
            X_new = res.x.reshape(N, 3)
            X_new = _project_to_sphere(X_new)
            return X_new
    except Exception:
        pass
    
    return X

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Start with Fibonacci spiral
    X = _fibonacci_sphere(N)
    X = _project_to_sphere(X)
    
    energy, grad = _energy_and_grad(X)
    best_X = X.copy()
    best_energy = energy
    
    # Phase 1: Riemannian conjugate gradient
    X_cg, energy_cg = _riemannian_cg(X, max_iter=500, tol=1e-12, 
                                       time_limit=60.0, start_time=start_time)
    if energy_cg < best_energy:
        best_energy = energy_cg
        best_X = X_cg.copy()
    
    # Phase 2: Try a few random restarts with RCG
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        n_restarts = min(5, int(remaining_time / 15))
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random perturbation
            X_pert = best_X + 0.05 * np.random.randn(N, 3)
            X_pert = _project_to_sphere(X_pert)
            
            e_pert, _ = _energy_and_grad(X_pert)
            
            # Only proceed if perturbation didn't increase energy too much
            if e_pert < best_energy + 5.0:
                X_cg_r, energy_cg_r = _riemannian_cg(X_pert, max_iter=200, tol=1e-11,
                                                       time_limit=12.0, start_time=time.time())
                if energy_cg_r < best_energy:
                    best_energy = energy_cg_r
                    best_X = X_cg_r.copy()
    
    # Phase 3: Final L-BFGS polish
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        X_final = _lbfgs_polish(best_X, time_limit=min(30.0, remaining_time - 2.0),
                                 start_time=start_time)
        e_final = _energy_only(X_final)
        if e_final < best_energy:
            best_energy = e_final
            best_X = X_final.copy()
    
    # Ensure output is valid
    result = best_X.copy()
    result = _project_to_sphere(result)
    
    # Check for very close points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(result[i] - result[j])
            if d < 1e-9:
                result[i] = result[i] + 1e-8 * np.random.randn(3)
                result[i] = result[i] / np.linalg.norm(result[i])
    
    return result
