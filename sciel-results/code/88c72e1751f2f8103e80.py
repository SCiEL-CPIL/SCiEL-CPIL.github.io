# EVOLVE-BLOCK-START
import numpy as np


def _broyden_solve(F, x_init, J_init=None, max_iter=25, tol=1e-12, eps=1e-7):
    """
    Broyden's method with line search.
    Returns (x, Fx, J, converged)
    """
    x = np.array(x_init, dtype=float)
    Fx = np.asarray(F(x), dtype=float)
    if not np.all(np.isfinite(Fx)):
        return x, Fx, J_init, False
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= tol:
        return x, Fx, J_init, True
    
    if J_init is None:
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            Fx_e = np.asarray(F(x + e), dtype=float)
            if not np.all(np.isfinite(Fx_e)):
                # Try smaller step
                eps2 = eps * 0.01
                e[j] = eps2
                Fx_e = np.asarray(F(x + e), dtype=float)
                if not np.all(np.isfinite(Fx_e)):
                    return x, Fx, J_init, False
                J[:, j] = (Fx_e - Fx) / eps2
            else:
                J[:, j] = (Fx_e - Fx) / eps
    else:
        J = J_init.copy()
    
    for _ in range(max_iter):
        if norm_Fx <= tol:
            break
        
        # Solve J * dx = -Fx
        try:
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            if abs(det) < 1e-14:
                dx = np.linalg.pinv(J) @ (-Fx)
            else:
                dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.pinv(J) @ (-Fx)
            except Exception:
                break
        
        if not np.all(np.isfinite(dx)):
            break
        
        # Line search with backtracking
        step = 1.0
        accepted = False
        for _ls in range(12):
            x_try = x + step * dx
            Fx_try = np.asarray(F(x_try), dtype=float)
            if np.all(np.isfinite(Fx_try)):
                norm_Fx_try = np.linalg.norm(Fx_try)
                # Armijo condition
                if norm_Fx_try <= norm_Fx - 1e-4 * step * np.linalg.norm(dx) * norm_Fx:
                    s = x_try - x
                    y = Fx_try - Fx
                    denom = np.dot(s, s)
                    if denom > 1e-14:
                        J = J + np.outer(y - J @ s, s) / denom
                    x = x_try
                    Fx = Fx_try
                    norm_Fx = norm_Fx_try
                    accepted = True
                    break
            step *= 0.5
        
        if not accepted:
            break
    
    converged = norm_Fx <= tol
    return x, Fx, J, converged


def _homotopy_solver(F, x0):
    """
    Homotopy/continuation method using Broyden's method with Jacobian reuse.
    Solves F_t(x) = t*F(x) + (1-t)*(x - x0) = 0 for t from 0 to 1.
    """
    x0 = np.array(x0, dtype=float)
    x = x0.copy()
    
    # At t=0, F_t(x) = x - x0. Root is x0.
    # Jacobian of F_t w.r.t x is t*J_F + (1-t)*I.
    # At t=0, J_t = I.
    J = np.eye(2)
    
    t = 0.0
    dt = 0.1
    max_dt = 0.5
    min_dt = 1e-4
    tol_t = 1e-12
    
    for _t_step in range(80):
        # Determine next t
        t_new = min(t + dt, 1.0)
        
        # Define the homotopy function for this t
        def F_t(x_val, _t=t_new):
            Fx = np.asarray(F(x_val), dtype=float)
            return _t * Fx + (1.0 - _t) * (x_val - x0)
        
        # Solve F_t(x) = 0 starting from current x
        # Use Broyden's method with current J as initial guess
        x_new, Fx_new, J, converged = _broyden_solve(F_t, x, J_init=J, max_iter=20, tol=tol_t)
        
        if converged:
            x = x_new
            t = t_new
            # Increase dt if successful
            if dt < max_dt:
                dt = min(dt * 1.5, max_dt)
            
            if t >= 1.0 - 1e-10:
                break
        else:
            # Failure, reduce dt
            dt *= 0.5
            if dt < min_dt:
                break
    
    # Check if we reached t=1 and converged
    if t >= 1.0 - 1e-10:
        F_final = np.asarray(F(x), dtype=float)
        if np.all(np.isfinite(F_final)) and np.linalg.norm(F_final) <= 1e-8:
            return x
    
    # If homotopy failed, try direct Broyden from current x
    x_direct, F_direct, _, _ = _broyden_solve(F, x, J_init=None, max_iter=30, tol=1e-13)
    if np.all(np.isfinite(F_direct)) and np.linalg.norm(F_direct) <= 1e-8:
        return x_direct
    
    # Try direct Broyden from x0 as well
    x_direct2, F_direct2, _, _ = _broyden_solve(F, x0, J_init=None, max_iter=30, tol=1e-13)
    if np.all(np.isfinite(F_direct2)) and np.linalg.norm(F_direct2) <= 1e-8:
        return x_direct2
    
    # Return best candidate
    F_x = np.asarray(F(x), dtype=float)
    F_xd = np.asarray(F(x_direct), dtype=float)
    F_xd2 = np.asarray(F(x_direct2), dtype=float)
    
    norm_x = np.linalg.norm(F_x) if np.all(np.isfinite(F_x)) else np.inf
    norm_xd = np.linalg.norm(F_xd) if np.all(np.isfinite(F_xd)) else np.inf
    norm_xd2 = np.linalg.norm(F_xd2) if np.all(np.isfinite(F_xd2)) else np.inf
    
    best_norm = min(norm_x, norm_xd, norm_xd2)
    if best_norm == norm_x:
        return x
    elif best_norm == norm_xd:
        return x_direct
    else:
        return x_direct2


def find_root_system(F, x0):
    """
    Wrapper for the homotopy solver.
    """
    x0 = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.zeros(2)
    
    x_result = _homotopy_solver(F, x0)
    
    # Final validation
    F_res = np.asarray(F(x_result), dtype=float)
    if np.all(np.isfinite(F_res)) and np.linalg.norm(F_res) <= 1e-8:
        return x_result
    
    return x_result
# EVOLVE-BLOCK-END
