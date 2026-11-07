# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with adaptive line search and fallback strategies.
    """
    x = np.array(x0, dtype=float).copy()
    if x.shape != (2,):
        x = x.reshape(2)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def time_up():
        return time.time() > deadline
    
    def norm(v):
        return float(np.linalg.norm(v))
    
    def is_valid(x):
        return x.shape == (2,) and np.all(np.isfinite(x))
    
    # Initial evaluation
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        Fx = np.array([1.0, 1.0])
    
    best_x = x.copy()
    best_norm = norm(Fx)
    
    if best_norm <= 1e-8:
        return best_x
    
    # Initial Jacobian via finite differences
    def compute_J(x, Fx):
        J = np.zeros((2, 2))
        h = 1e-7
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fp = F(x + e)
            if not np.all(np.isfinite(Fp)):
                Fp = Fx
            J[:, j] = (Fp - Fx) / h
        return J
    
    J = compute_J(x, Fx)
    
    # Broyden's method with adaptive line search
    max_iters = 50
    gamma = 1.0  # line search parameter
    
    for it in range(max_iters):
        if time_up():
            break
        
        # Check convergence
        if best_norm <= 1e-8:
            break
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # J is singular, use pseudo-inverse
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                # Last resort: steepest descent direction
                dx = -Fx
                dx_norm = norm(dx)
                if dx_norm > 0:
                    dx = dx / dx_norm * min(1.0, best_norm)
                else:
                    break
        
        # Limit step size to avoid huge jumps
        dx_norm = norm(dx)
        if dx_norm > 10.0:
            dx = dx / dx_norm * 10.0
            dx_norm = 10.0
        
        # Adaptive line search: try full step, then damped
        step = 1.0
        x_new = None
        Fx_new = None
        
        for ls_iter in range(10):
            x_trial = x + step * dx
            if not is_valid(x_trial):
                step *= 0.5
                continue
            
            Fx_trial = F(x_trial)
            if not np.all(np.isfinite(Fx_trial)):
                step *= 0.5
                continue
            
            n_trial = norm(Fx_trial)
            
            # Accept if we make sufficient progress (Armijo-like)
            if n_trial < best_norm * (1 - 0.01 * step * dx_norm / max(1.0, best_norm)):
                x_new = x_trial
                Fx_new = Fx_trial
                break
            
            step *= 0.5
        
        if x_new is None:
            # If no line search step accepted, try a small damped step
            step = 0.1
            x_trial = x + step * dx
            if is_valid(x_trial):
                Fx_trial = F(x_trial)
                if np.all(np.isfinite(Fx_trial)):
                    x_new = x_trial
                    Fx_new = Fx_trial
                else:
                    break
            else:
                break
        
        # Broyden update
        s = x_new - x
        y = Fx_new - Fx
        
        s_norm = norm(s)
        if s_norm < 1e-14:
            # No movement, break
            break
        
        # Update Jacobian: J_new = J + (y - J*s) * s^T / (s^T * s)
        J = J + np.outer(y - J @ s, s) / (s @ s)
        
        x = x_new
        Fx = Fx_new
        n = norm(Fx)
        
        if n < best_norm:
            best_norm = n
            best_x = x.copy()
        
        if best_norm <= 1e-8:
            break
    
    # If Broyden didn't converge, try a few restarts with different strategies
    if best_norm > 1e-8 and not time_up():
        # Restart 1: try from x0 with a slightly perturbed Jacobian
        x = np.array(x0, dtype=float).copy()
        Fx = F(x)
        if np.all(np.isfinite(Fx)):
            n = norm(Fx)
            if n < best_norm:
                best_norm = n
                best_x = x.copy()
            if best_norm <= 1e-8:
                return best_x
        
        # Use a different finite difference step
        J = np.zeros((2, 2))
        h = 1e-5
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fp = F(x + e)
            if np.all(np.isfinite(Fp)):
                J[:, j] = (Fp - Fx) / h
            else:
                J[:, j] = np.array([1.0, 0.0]) if j == 0 else np.array([0.0, 1.0])
        
        for it in range(30):
            if time_up():
                break
            if best_norm <= 1e-8:
                break
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                dx = -Fx
                dxn = norm(dx)
                if dxn > 0:
                    dx = dx / dxn * min(1.0, best_norm)
            
            dxn = norm(dx)
            if dxn > 5.0:
                dx = dx / dxn * 5.0
            
            x_new = x + dx
            if is_valid(x_new):
                Fx_new = F(x_new)
                if np.all(np.isfinite(Fx_new)):
                    n_new = norm(Fx_new)
                    s = x_new - x
                    y = Fx_new - Fx
                    sn = norm(s)
                    if sn > 1e-14:
                        J = J + np.outer(y - J @ s, s) / (s @ s)
                    x = x_new
                    Fx = Fx_new
                    if n_new < best_norm:
                        best_norm = n_new
                        best_x = x.copy()
    
    # Final check
    if best_norm <= 1e-8:
        return best_x
    
    # Return best point found
    return best_x
# EVOLVE-BLOCK-END
