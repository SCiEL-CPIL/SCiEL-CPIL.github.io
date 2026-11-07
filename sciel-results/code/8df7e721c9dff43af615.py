# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Robust Broyden's Method with line search and Jacobian re-estimation.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    Fx = np.array(F(x), dtype=float)
    calls = 1
    
    if np.linalg.norm(Fx) <= 1e-8:
        return x

    # Initialize Jacobian
    h = 1e-5
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        x_pert = x + e
        Fx_pert = np.array(F(x_pert), dtype=float)
        calls += 1
        J[:, j] = (Fx_pert - Fx) / h

    max_iter = 50
    for i in range(max_iter):
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break
        
        # Line search
        Fx_norm = np.linalg.norm(Fx)
        alpha = 1.0
        x_new = x + alpha * dx
        Fx_new = np.array(F(x_new), dtype=float)
        calls += 1
        Fx_new_norm = np.linalg.norm(Fx_new)
        
        if Fx_new_norm <= 1e-8:
            return x_new
        
        # Backtrack if no reduction
        if Fx_new_norm > Fx_norm:
            for _ in range(5):
                alpha *= 0.5
                x_new = x + alpha * dx
                Fx_new = np.array(F(x_new), dtype=float)
                calls += 1
                Fx_new_norm = np.linalg.norm(Fx_new)
                if Fx_new_norm <= 1e-8:
                    return x_new
                if Fx_new_norm < Fx_norm:
                    break
            else:
                # Failed to find reduction, re-estimate Jacobian
                h_fd = 1e-6
                J_new = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h_fd
                    x_pert = x + e
                    Fx_pert = np.array(F(x_pert), dtype=float)
                    calls += 1
                    J_new[:, j] = (Fx_pert - Fx) / h_fd
                J = J_new
                continue
        
        # Update Jacobian
        s = x_new - x
        y = Fx_new - Fx
        s_dot_s = np.dot(s, s)
        if s_dot_s > 1e-15:
            J = J + np.outer(y - J @ s, s) / s_dot_s
        
        x = x_new
        Fx = Fx_new

    return x
# EVOLVE-BLOCK-END
