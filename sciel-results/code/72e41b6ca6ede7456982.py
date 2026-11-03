import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _energy_and_grad(pts):
    """
    Compute the logarithmic energy E = sum_{i<j} log(1/||x_i - x_j||)
    and its Euclidean gradient.
    
    E = -sum_{i<j} log(||x_i - x_j||)
      = -0.5 * sum_{i<j} log(||x_i - x_j||^2)
    
    grad_E_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = pts.shape[0]
    # Squared distances
    # D2[i, j] = ||x_i - x_j||^2
    D2 = np.sum(pts**2, axis=1)[:, None] + np.sum(pts**2, axis=1)[None, :] - 2.0 * (pts @ pts.T)
    
    # Clip to avoid log(0) or division by zero
    D2 = np.maximum(D2, 1e-15)
    np.fill_diagonal(D2, 1.0)
    
    # Energy
    # Sum_{i<j} log(1/sqrt(D2_ij)) = -0.5 * sum_{i<j} log(D2_ij)
    # Sum_{i<j} log(D2_ij) = 0.5 * (Sum_{i != j} log(D2_ij))
    # So E = -0.25 * Sum_{i != j} log(D2_ij)
    log_D2 = np.log(D2)
    np.fill_diagonal(log_D2, 0.0)
    energy = -0.25 * np.sum(log_D2)
    
    # Gradient
    # grad_i = sum_{j != i} (x_i - x_j) / D2_ij
    # Let Diff[i, j, k] = pts[i, k] - pts[j, k]
    Diff = pts[:, None, :] - pts[None, :, :]
    
    # We need to divide each pair difference by D2_ij
    # D2 shape (N, N). We want to multiply Diff[i, j, :] by 1/D2[i, j]
    InvD2 = 1.0 / D2
    np.fill_diagonal(InvD2, 0.0)
    
    # Gradient component for point i is sum_j (x_i - x_j) / D2_ij
    # This is a matrix multiplication of Diff (N, N, 3) by InvD2 (N, N) along the j-axis
    grad = np.sum(Diff * InvD2[:, :, None], axis=1)
    
    return energy, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_init():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(X)

def _random_init(seed=None):
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.randn(N, 3)
    return _project_to_sphere(pts)

def _optimize_lbfgs(x0, time_budget, maxiter=2000, ftol=1e-14, gtol=1e-10):
    """
    Run L-BFGS-B optimization on the embedded space with projection.
    This uses the structure from the inspiration for a more robust local search.
    """
    start_time = time.time()
    
    def obj_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, g = _energy_and_grad(pts)
        # Project gradient to tangent space
        dot_g_x = np.sum(g * pts, axis=1, keepdims=True)
        g_tan = g - dot_g_x * pts
        return e, g_tan.flatten()
    
    x_flat = x0.flatten()
    best_x = x0.copy()
    best_e = _energy_and_grad(x0)[0]
    
    try:
        res = minimize(
            obj_and_grad,
            x_flat,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 2, 'ftol': ftol, 'gtol': gtol}
        )
        x_opt = _project_to_sphere(res.x.reshape(N, 3))
        e_opt = _energy_and_grad(x_opt)[0]
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
    except Exception:
        pass
    
    # Check if we have time for a refinement pass
    elapsed = time.time() - start_time
    if elapsed < time_budget * 0.5:
        try:
            res = minimize(
                obj_and_grad,
                best_x.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter, 'maxfun': maxiter * 2, 'ftol': ftol * 0.01, 'gtol': gtol * 0.01}
            )
            x_opt = _project_to_sphere(res.x.reshape(N, 3))
            e_opt = _energy_and_grad(x_opt)[0]
            if e_opt < best_e:
                best_e = e_opt
                best_x = x_opt.copy()
        except Exception:
            pass
            
    return best_x, best_e

def _optimize_gd(x0, deadline):
    """
    Run Gradient Descent with momentum and adaptive learning rate.
    This uses the structure from the current program.
    """
    X = x0.copy()
    V = np.zeros_like(X)
    lr = 0.01
    momentum = 0.9
    best_e, best_X = _energy_and_grad(X)[0], X.copy()
    
    # Calculate initial gradient
    e, g = _energy_and_grad(X)
    dot_g_x = np.sum(g * X, axis=1, keepdims=True)
    g_tan = g - dot_g_x * X
    
    for i in range(20000):
        if time.time() > deadline:
            break
        
        # Update velocity
        V = momentum * V + lr * g_tan
        X_new = X + V
        X_new = _project_to_sphere(X_new)
        
        # Compute new energy
        e_new, g_new = _energy_and_grad(X_new)
        
        # Simple line search
        alpha = 1.0
        while e_new > e + 1e-4 * alpha * np.sum(g_tan * V):
            alpha *= 0.5
            if alpha < 1e-6:
                break
            V_trial = momentum * V + lr * g_tan
            X_trial = X + alpha * V_trial
            X_trial = _project_to_sphere(X_trial)
            e_trial, _ = _energy_and_grad(X_trial)
            if e_trial < e:
                e_new = e_trial
                X_new = X_trial
                break
        
        # Accept step
        X = X_new
        e = e_new
        
        if e < best_e:
            best_e = e
            best_X = X.copy()
            
        # Update gradient
        e, g = _energy_and_grad(X)
        dot_g_x = np.sum(g * X, axis=1, keepdims=True)
        g_tan = g - dot_g_x * X
        
        # Decay learning rate
        lr *= 0.999
        
    return best_X, best_e

def _polish(x0, deadline):
    """
    Final high-precision polish pass using very small steps.
    """
    X = x0.copy()
    V = np.zeros_like(X)
    lr = 1e-4
    momentum = 0.9
    
    e, g = _energy_and_grad(X)
    dot_g_x = np.sum(g * X, axis=1, keepdims=True)
    g_tan = g - dot_g_x * X
    
    best_e = e
    best_X = X.copy()
    
    for i in range(5000):
        if time.time() > deadline:
            break
            
        V = momentum * V + lr * g_tan
        X_new = X + V
        X_new = _project_to_sphere(X_new)
        
        e_new, g_new = _energy_and_grad(X_new)
        
        # Simple line search
        alpha = 1.0
        while e_new > e + 1e-4 * alpha * np.sum(g_tan * V):
            alpha *= 0.5
            if alpha < 1e-6:
                break
            V_trial = momentum * V + lr * g_tan
            X_trial = X + alpha * V_trial
            X_trial = _project_to_sphere(X_trial)
            e_trial, _ = _energy_and_grad(X_trial)
            if e_trial < e:
                e_new = e_trial
                X_new = X_trial
                break
        
        X = X_new
        e = e_new
        
        if e < best_e:
            best_e = e
            best_X = X.copy()
            
        # Update gradient
        e, g = _energy_and_grad(X)
        dot_g_x = np.sum(g * X, axis=1, keepdims=True)
        g_tan = g - dot_g_x * X
        
        lr *= 0.99
        
    return best_X, best_e

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # 1. Initial guesses
    X_fib = _fibonacci_init()
    X_rand1 = _random_init(seed=1)
    X_rand2 = _random_init(seed=123)
    
    best_X = X_fib
    best_E = _energy_and_grad(X_fib)[0]
    
    candidates = [X_fib, X_rand1, X_rand2]
    
    # 2. Optimize each candidate using L-BFGS-B (more robust local search)
    for idx, X0 in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
        
        # Allocate time budget for this optimization
        # We have 3 candidates. Let's give them roughly equal time, but save some for polish.
        # Total time ~140s. Save 20s for polish. 120s / 3 = 40s each.
        sub_deadline = min(deadline - 20.0, start_time + 40.0 * (idx + 1))
        
        X_opt, E_opt = _optimize_lbfgs(X0, sub_deadline - time.time())
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
    
    # 3. Final polish with a more precise method if time permits
    if time.time() < deadline - 5.0:
        # Polish
        X_polish, E_polish = _polish(best_X, deadline - 1.0)
        
        if E_polish < best_E:
            best_E = E_polish
            best_X = X_polish
    
    # Final check for validity
    norms = np.linalg.norm(best_X, axis=1)
    if np.any(norms < 1e-9) or np.any(~np.isfinite(best_X)):
        return X_fib # Fallback
    
    return best_X
