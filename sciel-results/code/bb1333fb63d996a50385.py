# EVOLVE-BLOCK-START
import numpy as np


def _broyden_core(F, x, J, max_iter=40, tol=1e-13, eps=1e-7):
    """
    Core Broyden iteration. Returns (x, J, converged, n_calls).
    J is the current Jacobian estimate (will be updated in-place).
    """
    x = np.array(x, dtype=float)
    Fx = np.asarray(F(x), dtype=float)
    n_calls = 1
    if not np.all(np.isfinite(Fx)):
        return x, J, False, n_calls
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= tol:
        return x, J, True, n_calls
    
    for _ in range(max_iter):
        if norm_Fx <= tol:
            break
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
        
        step = 1.0
        accepted = False
        for _ in range(12):
            x_new = x + step * dx
            Fx_new = np.asarray(F(x_new), dtype=float)
            n_calls += 1
            if np.all(np.isfinite(Fx_new)):
                norm_Fx_new = np.linalg.norm(Fx_new)
                if norm_Fx_new <= norm_Fx - 1e-4 * step * np.linalg.norm(dx) * norm_Fx:
                    s = x_new - x
                    y = Fx_new - Fx
                    denom = np.dot(s, s)
                    if denom > 1e-14:
                        J = J + np.outer(y - J @ s, s) / denom
                    x = x_new
                    Fx = Fx_new
                    norm_Fx = norm_Fx_new
                    accepted = True
                    break
            step *= 0.5
        if not accepted:
            break
    
    return x, J, norm_Fx <= tol, n_calls


def _fd_jacobian(F, x, eps=1e-7):
    """Compute Jacobian via finite differences. Returns (J, n_calls)."""
    Fx = np.asarray(F(x), dtype=float)
    n_calls = 1
    if not np.all(np.isfinite(Fx)):
        return None, n_calls
    
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        Fx_e = np.asarray(F(x + e), dtype=float)
        n_calls += 1
        if not np.all(np.isfinite(Fx_e)):
            return None, n_calls
        J[:, j] = (Fx_e - Fx) / eps
    
    return J, n_calls


def _homotopy_solver(F, x0, max_iter=100, tol=1e-12, max_t_steps=50, min_dt=1e-4):
    """
    Homotopy/continuation method using Broyden's method with Jacobian reuse.
    Solves F_t(x) = t*F(x) + (1-t)*(x - x0) = 0 for t from 0 to 1.
    """
    x0 = np.array(x0, dtype=float)
    x = x0.copy()
    
    J = np.eye(2)
    
    t = 0.0
    dt = 0.1
    max_dt = 0.5
    
    for _ in range(max_t_steps):
        t_new = min(t + dt, 1.0)
        
        def F_t(x_val):
            Fx = np.asarray(F(x_val), dtype=float)
            return t_new * Fx + (1.0 - t_new) * (x_val - x0)
        
        x_new = x.copy()
        Fx_new = F_t(x_new)
        norm_Fx = np.linalg.norm(Fx_new)
        
        converged = False
        for _iter in range(25):
            if norm_Fx < tol:
                converged = True
                break
            
            try:
                det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                if abs(det) < 1e-14:
                    dx = np.linalg.pinv(J) @ (-Fx_new)
                else:
                    dx = np.linalg.solve(J, -Fx_new)
            except np.linalg.LinAlgError:
                dx = np.linalg.pinv(J) @ (-Fx_new)
            
            if not np.all(np.isfinite(dx)):
                break
            
            step = 1.0
            accepted = False
            for _ls in range(10):
                x_try = x_new + step * dx
                Fx_try = F_t(x_try)
                if np.all(np.isfinite(Fx_try)):
                    norm_Fx_try = np.linalg.norm(Fx_try)
                    if norm_Fx_try <= norm_Fx - 1e-4 * step * np.linalg.norm(dx) * norm_Fx:
                        s = x_try - x_new
                        y = Fx_try - Fx_new
                        denom = np.dot(s, s)
                        if denom > 1e-14:
                            J = J + np.outer(y - J @ s, s) / denom
                        x_new = x_try
                        Fx_new = Fx_try
                        norm_Fx = norm_Fx_try
                        accepted = True
                        break
                step *= 0.5
            
            if not accepted:
                break
            
            if norm_Fx < tol:
                converged = True
                break
        
        if converged:
            x = x_new
            t = t_new
            if dt < max_dt:
                dt = min(dt * 1.5, max_dt)
            
            if t >= 1.0 - 1e-10:
                break
        else:
            dt *= 0.5
            if dt < min_dt:
                break
    
    F_final = np.asarray(F(x), dtype=float)
    if np.all(np.isfinite(F_final)) and np.linalg.norm(F_final) <= 1e-8:
        return x
    
    x_direct, J_direct, conv_direct, _ = _broyden_core(F, x, J.copy(), max_iter=30, tol=1e-13)
    F_direct = np.asarray(F(x_direct), dtype=float)
    if np.all(np.isfinite(F_direct)) and np.linalg.norm(F_direct) <= 1e-8:
        return x_direct
    
    J_fresh, _ = _fd_jacobian(F, x0)
    if J_fresh is not None:
        x_direct2, _, conv2, _ = _broyden_core(F, x0, J_fresh, max_iter=30, tol=1e-13)
        F_direct2 = np.asarray(F(x_direct2), dtype=float)
        if np.all(np.isfinite(F_direct2)) and np.linalg.norm(F_direct2) <= 1e-8:
            return x_direct2
    
    norm_hom = np.linalg.norm(F_final) if np.all(np.isfinite(F_final)) else np.inf
    norm_dir = np.linalg.norm(F_direct) if np.all(np.isfinite(F_direct)) else np.inf
    
    if norm_dir < norm_hom:
        return x_direct
    return x


def find_root_system(F, x0):
    """
    Wrapper for the homotopy solver.
    """
    x0 = np.array(x0, dtype=float)
    # Ensure x0 is finite
    if not np.all(np.isfinite(x0)):
        x0 = np.zeros(2)
        
    # Try homotopy
    x_result = _homotopy_solver(F, x0)
    
    # Final validation
    F_res = np.asarray(F(x_result), dtype=float)
    if np.all(np.isfinite(F_res)) and np.linalg.norm(F_res) <= 1e-8:
        return x_result
    
    # If it failed, maybe try a simpler direct Broyden from x0?
    # The homotopy solver already tries a direct Broyden fallback internally.
    # But just in case, we can return the result from the homotopy solver's best effort.
    return x_result
# EVOLVE-BLOCK-END
