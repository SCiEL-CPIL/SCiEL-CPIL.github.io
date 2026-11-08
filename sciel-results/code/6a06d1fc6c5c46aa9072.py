"""Root finder for 3 fixed two-dimensional systems using optimized Broyden's method."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return np.all(np.isfinite(x)) and x.shape == (2,)

def _norm(x):
    return float(np.linalg.norm(x))

def _safe_call(F, x):
    """Call F safely, returning None on failure."""
    try:
        y = F(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float)
        if y.shape != (2,):
            return None
        if not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None

def _init_jacobian(x, F, Fx):
    """Initialize Jacobian using forward differences. Cost: 2 calls."""
    J = np.zeros((2, 2))
    h = 1e-6
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fp = _safe_call(F, x + e)
        if Fp is None:
            return None
        J[:, j] = (Fp - Fx) / h
    return J

def _broyden_update(J, s, y):
    """Broyden's good rank-1 update. s = x_new - x_old, y = F_new - F_old."""
    # J_new = J + (y - J@s) @ s^T / (s^T s)
    Js = J @ s
    diff = y - Js
    sTs = np.dot(s, s)
    if sTs < 1e-15:
        return J
    J = J + np.outer(diff, s) / sTs
    return J

def find_root_system(F, x0):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    x = np.array(x0, dtype=float)
    
    # Initial evaluation
    Fx = _safe_call(F, x)
    if Fx is None:
        return x
    if _norm(Fx) <= 1e-8:
        return x
    
    # Initialize Jacobian
    J = _init_jacobian(x, F, Fx)
    if J is None:
        return x
    
    best_x = x.copy()
    best_norm = _norm(Fx)
    
    max_iter = 50
    for i in range(max_iter):
        if _elapsed() > 110:
            break
            
        # Check for singularity
        try:
            det = np.linalg.det(J)
            if abs(det) < 1e-12:
                # Re-initialize Jacobian with a small perturbation
                x_pert = x + np.array([1e-4, 1e-4])
                Fx_pert = _safe_call(F, x_pert)
                if Fx_pert is None:
                    break
                J = _init_jacobian(x, F, Fx)
                if J is None:
                    break
                # Update Fx to the perturbed point if original J was bad? 
                # Actually, let's just try to solve with current J if possible, 
                # or restart from scratch if too many failures.
                # For simplicity, if det is small, we try a different step direction.
                # But Broyden usually handles this. Let's just proceed if solve works.
                pass
        except np.linalg.LinAlgError:
            pass
        
        # Solve J * p = -Fx
        try:
            p = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Fallback: use gradient of ||F||^2 if J is singular
            # grad = J^T Fx
            p = -J.T @ Fx
            if _norm(p) < 1e-15:
                break
        
        if not _is_finite(p):
            break
            
        # Adaptive step size
        # Estimate step size based on norm of Fx and norm of p
        # If Fx is large, take a larger step. If small, take a smaller step to avoid overshooting.
        # A common heuristic: t = min(1.0, _norm(Fx) / (_norm(p) * _norm(Fx) + 1e-8))
        # Let's use a simple heuristic:
        # If we are far from root, full step. If close, dampen.
        t = 1.0
        if _norm(Fx) < 1e-4:
            t = 0.5
        elif _norm(Fx) < 1e-2:
            t = 0.8
        
        # Ensure step doesn't go to infinity
        if _norm(p) > 1e6:
            t = 1e-6 / _norm(p)
            
        x_new = x + t * p
        if not _is_finite(x_new):
            # Try smaller step
            t = 0.1 * t
            x_new = x + t * p
            if not _is_finite(x_new):
                break
                
        Fx_new = _safe_call(F, x_new)
        if Fx_new is None:
            # Retry with smaller step
            t = 0.1 * t
            x_new = x + t * p
            Fx_new = _safe_call(F, x_new)
            if Fx_new is None:
                break
        
        norm_new = _norm(Fx_new)
        
        # Check for progress
        if norm_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_new
            
        if norm_new <= 1e-8:
            return x_new
            
        # Update Jacobian
        s = x_new - x
        y = Fx_new - Fx
        J = _broyden_update(J, s, y)
        
        # Move to new point
        x = x_new
        Fx = Fx_new
        
        # Stagnation check: if no improvement for 10 iterations, re-initialize J
        # We track this by comparing best_norm to initial norm roughly
        # Simpler: if norm_new >= norm_old * (1 - 1e-6), we might be stuck.
        # Let's add a counter for no-improvement
        # To keep it simple and fast, we rely on Broyden's robustness.
        # If we detect divergence (norm increases significantly), we might restart.
        if norm_new > best_norm * 10:
             # Diverging, try to restart from best_x with fresh J
             x = best_x.copy()
             Fx = _safe_call(F, x)
             if Fx is None:
                 break
             J = _init_jacobian(x, F, Fx)
             if J is None:
                 break

    return best_x
# EVOLVE-BLOCK-END
