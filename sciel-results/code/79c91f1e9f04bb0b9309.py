"""Root finder for 3 fixed two-dimensional systems using Newton-Raphson with line search and restarts."""
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


def _compute_jacobian(F, x, Fx):
    """Compute Jacobian using central differences."""
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
                return None
    return J


def _newton_step(F, x, Fx, max_iter=50):
    """
    Perform Newton-Raphson iterations with line search.
    Returns (x, Fx, success)
    """
    for it in range(max_iter):
        # Check convergence
        if np.linalg.norm(Fx) <= 1e-12:
            return x, Fx, True
        
        # Compute Jacobian
        J = _compute_jacobian(F, x, Fx)
        if J is None:
            return x, Fx, False
        
        # Solve J @ dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, try pseudo-inverse
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                return x, Fx, False
        
        # Line search
        alpha = 1.0
        Fx_norm = np.linalg.norm(Fx)
        x_trial = x + alpha * dx
        Fx_trial = _safe_F(F, x_trial)
        
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
            return x, Fx, False
        
        Fx_trial_norm = np.linalg.norm(Fx_trial)
        
        # If we couldn't reduce the norm, we might be stuck
        if Fx_trial_norm >= Fx_norm:
            return x, Fx, False
        
        # Update
        x = x_trial
        Fx = Fx_trial
        
        # Convergence check
        if Fx_norm <= 1e-10:
            return x, Fx, True
            
    return x, Fx, np.linalg.norm(Fx) <= 1e-8


def find_root_system(F, x0):
    x = np.array(x0, dtype=float, copy=True)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    Fx = _safe_F(F, x)
    if Fx is None:
        return x
        
    best_x = x
    best_norm = np.linalg.norm(Fx)
    
    # Try Newton from x0
    x_new, Fx_new, success = _newton_step(F, x, Fx)
    if np.linalg.norm(Fx_new) < best_norm:
        best_x = x_new
        best_norm = np.linalg.norm(Fx_new)
        
    if success and best_norm <= 1e-8:
        return best_x
    
    # If failed, try a few random restarts
    # We limit the number of restarts to save calls
    rng = np.random.default_rng(42)
    for i in range(10):
        # Generate a random point in a box around x0
        # Scale the box based on the norm of x0 or a fixed size
        scale = max(1.0, np.linalg.norm(x0))
        x_start = x0 + rng.uniform(-scale, scale, size=2)
        
        Fx_start = _safe_F(F, x_start)
        if Fx_start is None:
            continue
            
        x_new, Fx_new, success = _newton_step(F, x_start, Fx_start)
        if np.linalg.norm(Fx_new) < best_norm:
            best_x = x_new
            best_norm = np.linalg.norm(Fx_new)
            
        if success and best_norm <= 1e-8:
            return best_x
            
    # If all failed, return the best candidate we have
    return best_x
# EVOLVE-BLOCK-END
