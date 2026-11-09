"""Root finder for 3 fixed two-dimensional systems using a robust hybrid approach."""
# EVOLVE-BLOCK-START
import numpy as np


def _safe_F(F, x):
    """Call F safely, returning a finite array of length 2 or None on failure."""
    try:
        y = F(x)
        y = np.asarray(y, dtype=float)
        if y.shape != (2,) or not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None


def _broyden_update(J, x_new, x_old, F_new, F_old):
    """Rank-1 Broyden update."""
    s = x_new - x_old
    y = F_new - F_old
    denom = np.dot(y, s)
    if abs(denom) > 1e-14:
        J = J + np.outer(y - J @ s, s) / denom
    return J


def find_root_system(F, x0):
    x = np.array(x0, dtype=float, copy=True)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    # First, get F(x) and build initial Jacobian via finite differences
    Fx = _safe_F(F, x)
    if Fx is None:
        return x
    
    # If already at a root, return
    if np.linalg.norm(Fx) <= 1e-12:
        return x
    
    # Build initial Jacobian using central differences for better accuracy
    h = 1e-5
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fp = _safe_F(F, x + e)
        Fm = _safe_F(F, x - e)
        if Fp is not None and Fm is not None:
            J[:, j] = (Fp - Fm) / (2 * h)
        else:
            # Fallback to forward difference
            if Fp is not None:
                J[:, j] = (Fp - Fx) / h
            elif Fm is not None:
                J[:, j] = (Fx - Fm) / h
            else:
                J[:, j] = np.array([1.0, 0.0]) if j == 0 else np.array([0.0, 1.0])
    
    # Ensure J is not singular
    det = np.linalg.det(J)
    if abs(det) < 1e-12:
        # Add small regularization
        J += 1e-8 * np.eye(2)
    
    # Main iteration: Broyden's method with line search and damping
    max_iter = 200
    for it in range(max_iter):
        # Solve J @ dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break
        
        # Line search: find alpha such that ||F(x + alpha*dx)|| < ||F(x)||
        alpha = 1.0
        Fx_norm = np.linalg.norm(Fx)
        x_trial = x + alpha * dx
        Fx_trial = _safe_F(F, x_trial)
        
        # Backtracking line search
        max_ls = 20
        ls_count = 0
        while Fx_trial is None or np.linalg.norm(Fx_trial) > 0.9 * Fx_norm:
            alpha *= 0.5
            x_trial = x + alpha * dx
            Fx_trial = _safe_F(F, x_trial)
            ls_count += 1
            if ls_count > max_ls or alpha < 1e-12:
                break
        
        if Fx_trial is None:
            # If we can't evaluate, try a smaller step in a random direction
            break
        
        Fx_trial_norm = np.linalg.norm(Fx_trial)
        
        # Check convergence
        if Fx_trial_norm <= 1e-12:
            x = x_trial
            return x
        
        # Update Broyden Jacobian
        J = _broyden_update(J, x_trial, x, Fx_trial, Fx)
        
        # Accept the step
        x = x_trial
        Fx = Fx_trial
        Fx_norm = Fx_trial_norm
        
        # Additional convergence check
        if Fx_norm <= 1e-10:
            return x
    
    # If Broyden didn't converge, try a global optimization approach
    # Use scipy's minimize to find a point where ||F(x)||^2 is minimized
    try:
        from scipy.optimize import minimize
        
        def objective(x):
            Fx = _safe_F(F, x)
            if Fx is None:
                return 1e10
            return np.dot(Fx, Fx)
        
        result = minimize(objective, x, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-16})
        if result.success and np.all(np.isfinite(result.x)):
            Fx_final = _safe_F(F, result.x)
            if Fx_final is not None and np.linalg.norm(Fx_final) <= 1e-8:
                return result.x
    except Exception:
        pass
    
    return x
# EVOLVE-BLOCK-END
