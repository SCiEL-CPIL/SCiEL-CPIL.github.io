# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Secant-Step Hybrid root finder for 2D systems.
    
    Strategy:
    1. Initialize Jacobian via two-point secant difference (2 calls).
    2. Alternate between Broyden rank-1 updates and full finite-difference
       Jacobian recomputation every k iterations to correct drift.
    3. Use adaptive step control to ensure robustness.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    # Initial function evaluation
    Fx = np.array(F(x), dtype=float)
    calls = 1
    
    # If already at root
    if np.linalg.norm(Fx) <= 1e-12:
        return x
    
    # Initialize Jacobian via two-point secant difference
    # Use a small step size for the secant
    h = 1e-5
    x_perturbed = x.copy()
    x_perturbed[0] += h
    Fx_perturbed = np.array(F(x_perturbed), dtype=float)
    calls += 1
    
    # First column of Jacobian
    J = np.zeros((2, 2))
    J[:, 0] = (Fx_perturbed - Fx) / h
    
    # Second column
    x_perturbed2 = x.copy()
    x_perturbed2[1] += h
    Fx_perturbed2 = np.array(F(x_perturbed2), dtype=float)
    calls += 1
    J[:, 1] = (Fx_perturbed2 - Fx) / h
    
    # Check if Jacobian is singular or ill-conditioned
    try:
        cond = np.linalg.cond(J)
        if not np.isfinite(cond) or cond > 1e15:
            # Fallback to a simple damped Newton with finite differences
            # Try a different step size
            h2 = 1e-4
            J[:, 0] = (np.array(F(x + np.array([h2, 0])), dtype=float) - Fx) / h2
            calls += 1
            J[:, 1] = (np.array(F(x + np.array([0, h2])), dtype=float) - Fx) / h2
            calls += 1
    except:
        pass
    
    # Broyden's method with periodic re-anchoring
    max_iter = 50
    reanchor_freq = 5  # Recompute full Jacobian every 5 iterations
    
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
        if step_norm > 1.0:
            dx = dx / step_norm
        
        x_new = x + dx
        Fx_new = np.array(F(x_new), dtype=float)
        calls += 1
        
        # Check if we made progress
        Fx_new_norm = np.linalg.norm(Fx_new)
        
        if Fx_new_norm <= 1e-12:
            return x_new
        
        # If we didn't make progress, damp the step
        if Fx_new_norm > F_norm:
            # Try half step
            x_half = x + 0.5 * dx
            Fx_half = np.array(F(x_half), dtype=float)
            calls += 1
            if np.linalg.norm(Fx_half) < F_norm:
                x_new = x_half
                Fx_new = Fx_half
                dx = 0.5 * dx
            else:
                # Try quarter step
                x_quarter = x + 0.25 * dx
                Fx_quarter = np.array(F(x_quarter), dtype=float)
                calls += 1
                if np.linalg.norm(Fx_quarter) < F_norm:
                    x_new = x_quarter
                    Fx_new = Fx_quarter
                    dx = 0.25 * dx
                else:
                    # Backtrack failed, try a random perturbation
                    break
        
        # Update Jacobian using Broyden's rank-1 update
        s = x_new - x
        y = Fx_new - Fx
        
        # Broyden's good rank-1 update: J_new = J + (y - J*s) * s^T / (s^T * s)
        s_dot_s = np.dot(s, s)
        if s_dot_s > 1e-15:
            J = J + np.outer(y - J @ s, s) / s_dot_s
        
        x = x_new
        Fx = Fx_new
        
        # Periodic re-anchoring: recompute full Jacobian
        if (i + 1) % reanchor_freq == 0:
            h_fd = 1e-6
            Fx_current = Fx
            J_new = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h_fd
                Fx_pert = np.array(F(x + e), dtype=float)
                calls += 1
                J_new[:, j] = (Fx_pert - Fx_current) / h_fd
            J = J_new
    
    return x
# EVOLVE-BLOCK-END
