"""Root finder for 3 fixed 2D systems using Broyden's method with safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 using Broyden's method with line search.
    
    F: callable, takes array of shape (2,), returns array of shape (2,)
    x0: initial guess, array of shape (2,)
    Returns: x such that ||F(x)|| <= 1e-8
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    def time_up():
        return time.time() > deadline
    
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    # Initial evaluation
    try:
        Fx = F(x)
    except Exception:
        return x
    
    if not np.all(np.isfinite(Fx)):
        return x
    
    norm_F = np.linalg.norm(Fx)
    if norm_F <= 1e-12:
        return x
    
    # Initialize Jacobian approximation using finite differences
    # Use a slightly larger step for better conditioning
    h = 1e-5
    J = np.zeros((2, 2))
    Fx_orig = Fx.copy()
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        try:
            Fx_pert = F(x + e)
        except Exception:
            Fx_pert = Fx_orig + e * 1.0
        if not np.all(np.isfinite(Fx_pert)):
            Fx_pert = Fx_orig + e * 1.0
        J[:, j] = (Fx_pert - Fx_orig) / h
    
    # If Jacobian is singular or ill-conditioned, use identity
    try:
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-10:
            J = np.eye(2)
    except np.linalg.LinAlgError:
        J = np.eye(2)
    
    best_x = x.copy()
    best_norm = norm_F
    
    max_iter = 100
    for iteration in range(max_iter):
        if time_up():
            break
        
        # Check convergence
        if norm_F <= 1e-12:
            return x
        
        # Compute Newton step: J * delta = -Fx
        try:
            delta = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If solve fails, try pseudoinverse
            try:
                delta = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                # Fallback: gradient descent step
                delta = -Fx * 0.1
        
        if not np.all(np.isfinite(delta)):
            delta = -Fx * 0.1
        
        # Line search with backtracking
        step = 1.0
        x_new = x + step * delta
        Fx_new = None
        for _ in range(20):
            if time_up():
                break
            try:
                Fx_new = F(x_new)
            except Exception:
                step *= 0.5
                x_new = x + step * delta
                continue
            
            if not np.all(np.isfinite(Fx_new)):
                step *= 0.5
                x_new = x + step * delta
                continue
            
            norm_F_new = np.linalg.norm(Fx_new)
            
            # Armijo condition: sufficient decrease
            if norm_F_new <= (1 - 1e-4 * step) * norm_F:
                break
            
            step *= 0.5
            x_new = x + step * delta
        
        if Fx_new is None or not np.all(np.isfinite(Fx_new)):
            # If line search failed, try a smaller step
            step = 1e-4
            x_new = x + step * delta
            try:
                Fx_new = F(x_new)
            except Exception:
                break
            if not np.all(np.isfinite(Fx_new)):
                break
            norm_F_new = np.linalg.norm(Fx_new)
        else:
            norm_F_new = np.linalg.norm(Fx_new)
        
        # Update Broyden's method
        if np.all(np.isfinite(Fx_new)):
            # Broyden update: J_new = J + (Fx_new - Fx - J*delta) * delta^T / (delta^T * delta)
            delta_F = Fx_new - Fx
            J_delta = J @ delta
            correction = (delta_F - J_delta)
            denom = delta @ delta
            if denom > 1e-15:
                J = J + np.outer(correction, delta) / denom
            
            x = x_new
            Fx = Fx_new
            norm_F = norm_F_new
        
        # Track best solution
        if norm_F < best_norm:
            best_norm = norm_F
            best_x = x.copy()
        
        # Check for stagnation
        if norm_F > 1e3 * (1 + best_norm):
            # Diverging, try to reset
            x = best_x
            Fx = F(best_x)
            norm_F = np.linalg.norm(Fx)
            J = np.eye(2)
    
    # Final check
    if norm_F <= 1e-8:
        return x
    
    # Try a few more Newton steps with fresh Jacobian if close
    if best_norm < 1e-4 and not time_up():
        x = best_x
        for _ in range(10):
            if time_up():
                break
            try:
                Fx = F(x)
            except Exception:
                break
            if not np.all(np.isfinite(Fx)):
                break
            norm_F = np.linalg.norm(Fx)
            if norm_F <= 1e-12:
                return x
            
            # Recompute Jacobian
            h = 1e-5
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                try:
                    Fx_pert = F(x + e)
                except Exception:
                    Fx_pert = Fx + e
                if not np.all(np.isfinite(Fx_pert)):
                    Fx_pert = Fx + e
                J[:, j] = (Fx_pert - Fx) / h
            
            try:
                delta = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            if not np.all(np.isfinite(delta)):
                break
            
            x_new = x + delta
            try:
                Fx_new = F(x_new)
            except Exception:
                break
            if not np.all(np.isfinite(Fx_new)):
                break
            
            norm_F_new = np.linalg.norm(Fx_new)
            if norm_F_new < norm_F:
                x = x_new
                Fx = Fx_new
                norm_F = norm_F_new
                if norm_F <= 1e-12:
                    return x
            else:
                break
    
    return x if np.all(np.isfinite(x)) else best_x
# EVOLVE-BLOCK-END
