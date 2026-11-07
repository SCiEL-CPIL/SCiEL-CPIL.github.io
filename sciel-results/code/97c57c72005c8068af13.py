# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with adaptive damping and a time guard.
    """
    # Ensure x0 is a float numpy array
    x = np.array(x0, dtype=float).reshape(2)
    best_x = x.copy()
    best_norm = np.inf
    
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s kill
    
    def time_up():
        return time.time() > deadline
    
    def norm(v):
        return np.linalg.norm(v)
    
    def safe_F(x_in):
        """Wrapper to ensure F returns a valid numpy array"""
        try:
            res = F(x_in)
            res = np.array(res, dtype=float)
            if res.shape != (2,):
                # Reshape if possible, otherwise return invalid marker
                if res.size == 2:
                    res = res.reshape(2)
                else:
                    return None
            if not np.all(np.isfinite(res)):
                return None
            return res
        except Exception:
            return None

    try:
        # Initial function evaluation
        Fx = safe_F(x)
        if Fx is None:
            return best_x
        
        current_norm = norm(Fx)
        if current_norm <= 1e-8:
            return x
        best_norm = current_norm

        # Initialize Jacobian approximation using finite differences
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            x_pert = x + e
            Fx_e = safe_F(x_pert)
            if Fx_e is None:
                Fx_e = Fx # Fallback to current value if perturbation fails
            J[:, j] = (Fx_e - Fx) / h
        
        # Broyden's method iteration
        max_iter = 100
        
        for it in range(max_iter):
            if time_up():
                break
            
            # Check if Jacobian is singular or ill-conditioned
            try:
                # Solve J * s = -Fx for step s
                s = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # If singular, try a damped step in direction of -Fx
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
                Fx_new = safe_F(x_new)
                
                if Fx_new is None:
                    lam *= 0.5
                    continue
                
                Fx_new_norm = norm(Fx_new)
                
                if Fx_new_norm <= 1e-8:
                    x = x_new
                    Fx = Fx_new
                    best_x = x.copy()
                    best_norm = Fx_new_norm
                    success = True
                    break
                
                # Check if we made progress compared to global best
                if Fx_new_norm < best_norm:
                    best_x = x_new.copy()
                    best_norm = Fx_new_norm
                
                # Armijo-like condition: require sufficient decrease from current
                # We use best_norm here to ensure monotonic decrease towards global best
                if Fx_new_norm < best_norm * (1 - 1e-4):
                    # Update Jacobian using Broyden's formula
                    s_actual = x_new - x
                    Fx_diff = Fx_new - Fx
                    
                    # Broyden rank-1 update
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
                # If damping didn't help, try a steepest descent step
                s_descent = -Fx
                s_norm = norm(s_descent)
                if s_norm < 1e-15:
                    break
                
                s_descent = s_descent / s_norm * 0.1
                
                x_new = x + s_descent
                Fx_new = safe_F(x_new)
                
                if Fx_new is None:
                    break
                
                Fx_new_norm = norm(Fx_new)
                
                if Fx_new_norm < best_norm:
                    # Save old values for Jacobian update
                    x_old = x
                    Fx_old = Fx
                    
                    x = x_new
                    Fx = Fx_new
                    best_x = x.copy()
                    best_norm = Fx_new_norm
                    
                    # Update Jacobian
                    s_actual = x - x_old
                    Fx_diff = Fx - Fx_old
                    denom = np.dot(s_actual, s_actual)
                    if denom > 1e-30:
                        J = J + np.outer(Fx_diff - J @ s_actual, s_actual) / denom
                else:
                    # No progress, break
                    break
            
            if best_norm <= 1e-8:
                break
                
    except Exception:
        # If any unexpected error occurs, return the best valid point found so far
        pass

    return best_x
# EVOLVE-BLOCK-END
