# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Secant-Step Hybrid root finder for 2D systems.
    
    Strategy:
    1. Initialize Jacobian via forward finite differences (2 calls).
    2. Alternate between Broyden rank-1 updates and full finite-difference
       Jacobian recomputation every k iterations to correct drift.
    3. Use adaptive step control with robust line search.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    # Initial function evaluation
    Fx = np.array(F(x), dtype=float)
    calls = 1
    
    # If already at root
    F_norm = np.linalg.norm(Fx)
    if F_norm <= 1e-12:
        return x
    
    # Initialize Jacobian via forward finite differences
    # Use adaptive step size based on the scale of x
    x_scale = max(np.max(np.abs(x)), 1.0)
    h = 1e-6 * x_scale
    
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_pert = np.array(F(x + e), dtype=float)
        calls += 1
        J[:, j] = (Fx_pert - Fx) / h
    
    # Check if Jacobian is singular or ill-conditioned; if so, try larger step
    try:
        cond = np.linalg.cond(J)
        if not np.isfinite(cond) or cond > 1e12:
            h2 = 1e-4 * x_scale
            for j in range(2):
                e = np.zeros(2)
                e[j] = h2
                Fx_pert = np.array(F(x + e), dtype=float)
                calls += 1
                J[:, j] = (Fx_pert - Fx) / h2
    except:
        pass
    
    # Broyden's method with periodic re-anchoring
    max_iter = 40
    reanchor_freq = 4  # Recompute full Jacobian every 4 iterations
    
    for i in range(max_iter):
        # Check convergence
        F_norm = np.linalg.norm(Fx)
        if F_norm <= 1e-12:
            return x
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break
        
        # Adaptive step control: damp if step is too large
        step_norm = np.linalg.norm(dx)
        if step_norm > 10.0:
            dx = dx / step_norm * 10.0
        
        x_new = x + dx
        Fx_new = np.array(F(x_new), dtype=float)
        calls += 1
        
        # Check if we made progress
        Fx_new_norm = np.linalg.norm(Fx_new)
        
        if Fx_new_norm <= 1e-12:
            return x_new
        
        # If we didn't make progress, use backtracking line search
        if Fx_new_norm > F_norm:
            alpha = 1.0
            for _ in range(5):
                alpha *= 0.5
                x_trial = x + alpha * dx
                Fx_trial = np.array(F(x_trial), dtype=float)
                calls += 1
                Fx_trial_norm = np.linalg.norm(Fx_trial)
                if Fx_trial_norm < F_norm:
                    x_new = x_trial
                    Fx_new = Fx_trial
                    break
            else:
                # Backtrack failed completely
                break
        
        # Update Jacobian using Broyden's good rank-1 update
        s = x_new - x
        y = Fx_new - Fx
        
        s_dot_s = np.dot(s, s)
        if s_dot_s > 1e-18:
            J = J + np.outer(y - J @ s, s) / s_dot_s
        
        x = x_new
        Fx = Fx_new
        
        # Periodic re-anchoring: recompute full Jacobian
        if (i + 1) % reanchor_freq == 0:
            x_scale = max(np.max(np.abs(x)), 1.0)
            h_fd = 1e-6 * x_scale
            J_new = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h_fd
                Fx_pert = np.array(F(x + e), dtype=float)
                calls += 1
                J_new[:, j] = (Fx_pert - Fx) / h_fd
            J = J_new
    
    return x
# EVOLVE-BLOCK-END
