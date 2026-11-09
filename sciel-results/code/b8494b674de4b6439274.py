"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 such that ||F(x)|| <= 1e-8 using Broyden's method
    with adaptive damping and a deadline guard.
    """
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    x = np.array(x0, dtype=float, copy=True)
    n = 2
    
    # Initial function evaluation
    Fx = np.array(F(x), dtype=float)
    
    if not np.all(np.isfinite(Fx)):
        return x
    
    # Initial Jacobian via finite differences
    h = 1e-7
    J = np.zeros((n, n))
    for j in range(n):
        xp = x.copy()
        xp[j] += h
        Fxp = np.array(F(xp), dtype=float)
        J[:, j] = (Fxp - Fx) / h
    
    # Check initial
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    
    # Broyden's method with adaptive damping
    best_x = x.copy()
    best_norm = norm_Fx
    damping = 1.0
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Solve J dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, try least squares
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
        
        # Adaptive damping: reduce step if it's too large
        dx_norm = np.linalg.norm(dx)
        if dx_norm > 1.0:
            dx = dx / dx_norm
        
        # Try full step first
        x_trial = x + damping * dx
        Fx_trial = np.array(F(x_trial), dtype=float)
        
        if not np.all(np.isfinite(Fx_trial)):
            # Non-finite, reduce damping
            damping *= 0.5
            continue
        
        norm_Fx_trial = np.linalg.norm(Fx_trial)
        
        if norm_Fx_trial <= 1e-8:
            return x_trial
        
        # Check if this step improves
        if norm_Fx_trial < best_norm:
            # Accept step
            best_x = x_trial.copy()
            best_norm = norm_Fx_trial
            
            # Update Broyden's formula
            delta_F = Fx_trial - Fx
            delta_x = x_trial - x
            J = J + np.outer(delta_F - J @ delta_x, delta_x) / np.dot(delta_x, delta_x)
            
            x = x_trial
            Fx = Fx_trial
            norm_Fx = norm_Fx_trial
            
            # Reset damping on successful step
            damping = 1.0
        else:
            # Reject step, reduce damping
            damping *= 0.5
            if damping < 1e-10:
                # Try a different direction: steepest descent on ||F||^2
                # Gradient of ||F||^2/2 is J^T F
                grad = J.T @ Fx
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-15:
                    dx = -grad / grad_norm
                    x_trial = x + 0.1 * dx
                    Fx_trial = np.array(F(x_trial), dtype=float)
                    if np.all(np.isfinite(Fx_trial)):
                        norm_Fx_trial = np.linalg.norm(Fx_trial)
                        if norm_Fx_trial < best_norm:
                            best_x = x_trial.copy()
                            best_norm = norm_Fx_trial
                            delta_F = Fx_trial - Fx
                            delta_x = x_trial - x
                            J = J + np.outer(delta_F - J @ delta_x, delta_x) / max(np.dot(delta_x, delta_x), 1e-30)
                            x = x_trial
                            Fx = Fx_trial
                            norm_Fx = norm_Fx_trial
                            damping = 1.0
                break
        
        # Early termination if close enough
        if norm_Fx <= 1e-10:
            break
    
    # Final check: if we haven't met the tolerance, try a few more Newton steps
    # with the current Jacobian estimate
    for _ in range(20):
        if time.time() > deadline:
            break
        
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x
        
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
        
        x_trial = x + dx
        Fx_trial = np.array(F(x_trial), dtype=float)
        
        if not np.all(np.isfinite(Fx_trial)):
            break
        
        norm_Fx_trial = np.linalg.norm(Fx_trial)
        
        if norm_Fx_trial < norm_Fx:
            delta_F = Fx_trial - Fx
            delta_x = x_trial - x
            J = J + np.outer(delta_F - J @ delta_x, delta_x) / max(np.dot(delta_x, delta_x), 1e-30)
            x = x_trial
            Fx = Fx_trial
        else:
            break
    
    # Return the best solution found
    return best_x
# EVOLVE-BLOCK-END
