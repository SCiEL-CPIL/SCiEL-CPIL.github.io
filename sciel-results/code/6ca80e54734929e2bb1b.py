"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
import math
import random

DEADLINE_FRACTION = 0.90
TIME_LIMIT = 150.0

def _energy_and_grad(X):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    and its gradient w.r.t. X (shape (N,3)).
    """
    dots = X @ X.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    
    iu = np.triu_indices(N, 1)
    log_d2 = np.log(d2[iu])
    E = -0.5 * math.fsum(log_d2.tolist())
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    row_sums = inv_d2.sum(axis=1)
    term2 = inv_d2 @ X
    # Gradient of sum log(d) is X * row_sums - term2
    # We minimize -sum log(d), so grad is -(X * row_sums - term2)
    # Wait, let's re-verify signs.
    # E = - sum log d.
    # dE/dx_i = - sum_{j != i} (x_i - x_j) / d^2
    # Current code: grad = X * row_sums - term2 = sum (x_i - x_j)/d^2.
    # So current grad is d(sum log d)/dx.
    # We want to minimize E, so we need -grad_current.
    # The current optimizer uses direction = -g.
    # If g is d(sum log d)/dx, then -g is d(-sum log d)/dx.
    # So the current code is correct for minimizing E.
    
    grad = X * row_sums[:, None] - term2
    
    return E, grad

def _hessian(X):
    """
    Compute the Hessian of the energy on the sphere.
    Returns a (3N, 3N) matrix.
    """
    H = np.zeros((3*N, 3*N))
    dots = X @ X.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    
    for i in range(N):
        for j in range(i+1, N):
            d_ij_sq = d2[i, j]
            if d_ij_sq < 1e-12: continue
            d_ij = np.sqrt(d_ij_sq)
            u = X[i] - X[j]
            
            # Hessian of -log(d) for pair (i,j) in R^3
            # f(u) = -log(||u||)
            # grad_f = -u / ||u||^2
            # hess_f = I/||u||^2 - 2 u u^T / ||u||^4
            
            # Let A = I/d^2 - 2 u u^T / d^4
            # H_ii += P_i A P_i
            # H_jj += P_j A P_j
            # H_ij += - P_i A P_j
            # H_ji += - P_j A P_i
            
            I3 = np.eye(3)
            A = I3 / d_ij_sq - 2.0 * np.outer(u, u) / (d_ij_sq * d_ij_sq)
            
            # Projection P_i = I - x_i x_i^T
            P_i = I3 - np.outer(X[i], X[i])
            P_j = I3 - np.outer(X[j], X[j])
            
            # Blocks
            H_ii_block = P_i @ A @ P_i
            H_jj_block = P_j @ A @ P_j
            H_ij_block = - (P_i @ A @ P_j)
            
            # Fill H
            H[3*i:3*i+3, 3*i:3*i+3] += H_ii_block
            H[3*j:3*j+3, 3*j:3*j+3] += H_jj_block
            H[3*i:3*i+3, 3*j:3*j+3] += H_ij_block
            H[3*j:3*j+3, 3*i:3*i+3] += H_ij_block.T
            
    return H

def _retract(X, step):
    """Riemannian retraction."""
    step_dot_X = np.sum(step * X, axis=1, keepdims=True)
    tangent_step = step - step_dot_X * X
    X_new = X + tangent_step
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    X_new = X_new / norms
    return X_new

def _fibonacci_sphere(n, seed_offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + seed_offset
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate n random points on the sphere."""
    theta = rng.uniform(0, 2 * np.pi, size=n)
    cos_phi = rng.uniform(-1, 1, size=n)
    sin_phi = np.sqrt(1 - cos_phi ** 2)
    x = sin_phi * np.cos(theta)
    y = sin_phi * np.sin(theta)
    z = cos_phi
    return np.column_stack([x, y, z])

def _optimize_lbfgs(X0, max_iter=2000, tol=1e-10, deadline=None):
    """
    L-BFGS optimization on the sphere.
    """
    X = X0.copy()
    n = X.shape[0]
    m = 20
    s_list = []
    y_list = []
    
    E, g = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        g_flat = g.ravel()
        g_norm = np.linalg.norm(g_flat)
        if g_norm < tol:
            break
        
        if len(s_list) == 0:
            direction = -g
        else:
            q = g_flat.copy()
            alphas = []
            rhos = []
            
            for i in range(len(s_list) - 1, -1, -1):
                s_i = s_list[i]
                y_i = y_list[i]
                sy = np.dot(s_i, y_i)
                if abs(sy) < 1e-15:
                    rho_i = 0.0
                else:
                    rho_i = 1.0 / sy
                rhos.append(rho_i)
                alpha_i = rho_i * np.dot(s_i, q)
                alphas.append(alpha_i)
                q = q - alpha_i * y_i
            
            r = q
            
            for i in range(len(s_list)):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[len(s_list) - 1 - i]
                alpha_i = alphas[len(s_list) - 1 - i]
                beta_i = rho_i * np.dot(y_i, r)
                r = r + (alpha_i - beta_i) * s_i
            
            direction = r.reshape(n, 3)
        
        # Ensure descent direction
        g_flat_dir = np.dot(g_flat, direction.ravel())
        if g_flat_dir >= 0:
            direction = -g
            g_flat_dir = np.dot(g_flat, direction.ravel())
        
        c1 = 1e-4
        c2 = 0.9
        step_size = 1.0
        
        X_trial = _retract(X, step_size * direction)
        E_trial, g_trial = _energy_and_grad(X_trial)
        
        max_ls = 30
        ls_success = False
        for ls in range(max_ls):
            if deadline is not None and time.time() > deadline:
                break
            armijo_ok = E_trial <= E + c1 * step_size * g_flat_dir
            g_trial_dir = np.dot(g_trial.ravel(), direction.ravel())
            curvature_ok = g_trial_dir >= c2 * g_flat_dir
            
            if armijo_ok and curvature_ok:
                ls_success = True
                break
            
            step_size *= 0.5
            X_trial = _retract(X, step_size * direction)
            E_trial, g_trial = _energy_and_grad(X_trial)
        
        if step_size < 1e-15:
            break
        
        if not ls_success and E_trial > E:
            break
        
        s_vec = (X_trial - X).ravel()
        y_vec = (g_trial - g).ravel()
        
        sy = np.dot(s_vec, y_vec)
        if sy > 1e-12:
            s_list.append(s_vec)
            y_list.append(y_vec)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        
        X = X_trial
        E = E_trial
        g = g_trial
        
        if E < best_E:
            best_E = E
            best_X = X.copy()
    
    return best_X, best_E

def _newton_polish(X0, max_iter=50, deadline=None):
    """
    Newton-Raphson polish using the Hessian.
    """
    X = X0.copy()
    E, g = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        H = _hessian(X)
        g_flat = g.ravel()
        
        # Solve H d = -g
        try:
            d = np.linalg.solve(H, -g_flat)
        except np.linalg.LinAlgError:
            d = np.linalg.lstsq(H, -g_flat, rcond=None)[0]
            
        # Project to tangent space
        d_reshaped = d.reshape(N, 3)
        d_dot_X = np.sum(d_reshaped * X, axis=1, keepdims=True)
        d_tangent = d_reshaped - d_dot_X * X
        
        # Line search for Newton step
        step_size = 1.0
        X_trial = _retract(X, step_size * d_tangent)
        E_trial, _ = _energy_and_grad(X_trial)
        
        for ls in range(20):
            if E_trial < E:
                X = X_trial
                E = E_trial
                g = _energy_and_grad(X)[1]
                if E < best_E:
                    best_E = E
                    best_X = X.copy()
                break
            step_size *= 0.5
            X_trial = _retract(X, step_size * d_tangent)
            E_trial, _ = _energy_and_grad(X_trial)
            
        if step_size < 1e-10:
            break
            
    return best_X, best_E

def construct():
    global _start_time
    _start_time = time.time()
    deadline = _start_time + TIME_LIMIT * DEADLINE_FRACTION
    
    rng = np.random.RandomState(42)
    
    starts = []
    # Fibonacci
    for offset in [0.0, 0.5, 1.0]:
        starts.append(_fibonacci_sphere(N, seed_offset=offset))
    
    # Random
    for _ in range(3):
        starts.append(_random_sphere(N, rng))
        
    # Perturbed Fibonacci
    for _ in range(3):
        base = _fibonacci_sphere(N)
        noise = rng.normal(0, 0.05, size=(N, 3))
        perturbed = base + noise
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / np.maximum(norms, 1e-15)
        starts.append(perturbed)

    best_pts = None
    best_energy = float('inf')
    
    for i, start in enumerate(starts):
        if deadline is not None and time.time() > deadline:
            break
        
        time_per_start = (deadline - time.time()) / max(1, len(starts) - i)
        start_deadline = time.time() + time_per_start
        
        pts, e = _optimize_lbfgs(start, max_iter=1000, tol=1e-10, deadline=start_deadline)
        
        # Newton Polish
        pts, e = _newton_polish(pts, max_iter=20, deadline=start_deadline)
        
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    if best_pts is not None:
        # Final tight polish
        best_pts, best_energy = _newton_polish(best_pts, max_iter=10, deadline=deadline)
        
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        best_pts = best_pts / norms
        
        if not np.all(np.isfinite(best_pts)):
            return _fibonacci_sphere(N)
            
        min_d = 1e18
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < min_d:
                    min_d = d
        if min_d < 1e-9:
            return _fibonacci_sphere(N)
            
    return best_pts
# EVOLVE-BLOCK-END
