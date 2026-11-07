# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with adaptive damping and a time guard.
    """
    x = np.array(x0, dtype=float).reshape(2)
    
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s kill
    
    def time_up():
        return time.time() > deadline
    
    def norm(v):
        return np.linalg.norm(v)
    
    # Initial function evaluation
    Fx = F(x)
    if not np.all(np.isfinite(Fx)) or Fx.shape != (2,):
        # F returned something invalid; return x as best effort
        return x
    
    if norm(Fx) <= 1e-8:
        return x
    
    # Initialize Jacobian approximation using finite differences
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_e = F(x + e)
        if not np.all(np.isfinite(Fx_e)):
            Fx_e = Fx
        J[:, j] = (Fx_e - Fx) / h
    
    # Broyden's method iteration
    max_iter = 100
    best_x = x.copy()
    best_norm = norm(Fx)
    
    for it in range(max_iter):
        if time_up():
            break
        
        # Check if Jacobian is singular or ill-conditioned
        try:
            # Solve J * s = -Fx for step s
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, try a damped step
            s = -Fx * 0.1
            if not np.all(np.isfinite(s)):
                break
        
        # Adaptive damping: start with full step, reduce if needed
        lam = 1.0
        max_damp = 10
        success = False
        
        for damp_iter in range(max_damp):
            if time_up():
                break
            
            x_new = x + lam * s
            Fx_new = F(x_new)
            
            if not np.all(np.isfinite(Fx_new)) or Fx_new.shape != (2,):
                lam *= 0.5
                continue
            
            Fx_new_norm = norm(Fx_new)
            
            if Fx_new_norm <= 1e-8:
                x = x_new
                Fx = Fx_new
                success = True
                break
            
            # Check if we made progress
            if Fx_new_norm < best_norm:
                best_x = x_new.copy()
                best_norm = Fx_new_norm
            
            # Armijo-like condition: require sufficient decrease
            if Fx_new_norm < best_norm * (1 - 1e-4):
                # Update Jacobian using Broyden's formula
                s_actual = x_new - x
                Fx_diff = Fx_new - Fx
                
                # Broyden rank-1 update: J = J + (Fx_diff - J @ s_actual) @ s_actual^T / (s_actual^T @ s_actual)
                denom = np.dot(s_actual, s_actual)
                if denom > 1e-30:
                    J = J + np.outer(Fx_diff - J @ s_actual, s_actual) / denom
                
                x = x_new
                Fx = Fx_new
                success = True
                break
            else:
                # No sufficient decrease, try smaller step
                lam *= 0.5
        
        if time_up():
            break
        
        if not success:
            # If damping didn't help, try a different strategy
            # Use a steepest descent step
            s_descent = -Fx
            s_norm = norm(s_descent)
            if s_norm < 1e-15:
                break
            
            s_descent = s_descent / s_norm * 0.1
            
            x_new = x + s_descent
            Fx_new = F(x_new)
            
            if not np.all(np.isfinite(Fx_new)) or Fx_new.shape != (2,):
                break
            
            Fx_new_norm = norm(Fx_new)
            
            if Fx_new_norm < best_norm:
                x = x_new
                Fx = Fx_new
                best_x = x.copy()
                best_norm = Fx_new_norm
                
                # Update Jacobian
                s_actual = x_new - x
                Fx_diff = Fx_new - Fx
                denom = np.dot(s_actual, s_actual)
                if denom > 1e-30:
                    J = J + np.outer(Fx_diff - J @ s_actual, s_actual) / denom
            else:
                # No progress, break
                break
        
        if best_norm <= 1e-8:
            break
    
    # Return the best point found
    return best_x
# EVOLVE-BLOCK-END
