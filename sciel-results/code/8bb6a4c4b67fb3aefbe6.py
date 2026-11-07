"""Homotopy-based root finder for 3 fixed 2D systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = None
_DEADLINE_MARGIN = 5  # seconds to spare before 150s limit

def _elapsed():
    return time.time() - _START_TIME if _START_TIME is not None else 0

def _time_left():
    return 150.0 - _elapsed() - _DEADLINE_MARGIN

def _safe_norm_F(x, F):
    """Compute ||F(x)||, return inf on error."""
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return np.inf
        return float(np.linalg.norm(Fx))
    except Exception:
        return np.inf

def _safe_F(x, F):
    """Compute F(x), return None on error."""
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return None
        return Fx
    except Exception:
        return None

def _newton_polish(F, x0, max_iter=50):
    """
    High-efficiency Newton-Raphson polish.
    Uses forward differences for Jacobian.
    Stops immediately when ||F(x)|| <= 1e-8.
    """
    x = np.array(x0, dtype=float)
    h = 1e-6  # Step size for FD Jacobian
    
    for i in range(max_iter):
        if _time_left() < 0.5:
            return x
        
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
                return x
            
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx <= 1e-8:
                return x
            
            # Compute Jacobian using forward differences
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_p = F(x + e)
                Fx_p = np.asarray(Fx_p, dtype=float)
                if Fx_p.shape != (2,) or not np.all(np.isfinite(Fx_p)):
                    return x
                J[:, j] = (Fx_p - Fx) / h
            
            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Fallback: gradient descent step
                dx = -0.1 * Fx
                
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                return x
            x = x_new
            
        except Exception:
            return x
            
    return x

def _homotopy_step(F, x_t, t_current, t_target, max_backtrack=20):
    """
    Attempt to move from t_current to t_target along the homotopy path.
    H(x, t) = (1-t)F(x) + t(x - x0)
    We want H(x, t) = 0.
    
    Returns (success, x_new, t_new)
    """
    x0 = x_t.copy() # This is actually the anchor point x0 of the system, not the current x
    # Wait, the homotopy definition needs the anchor x0 (initial guess) and current x.
    # Let's redefine: H(x, t) = (1-t)F(x) + t(x - x_anchor)
    # We are at x_t which is a root of H(x, t_current).
    # We want to find x_new such that H(x_new, t_target) = 0.
    
    # We use Newton's method on the extended system? No, just standard Newton on x for fixed t?
    # Better: Predictor-Corrector.
    # 1. Predictor: Linear extrapolation
    #    H(x, t) = 0 => (1-t)F(x) + t(x - x_anchor) = 0
    #    Differentiate w.r.t t: (1-t) J(x) dx/dt + (x - x_anchor) = 0
    #    dx/dt = - (1-t) J(x)^{-1} (x - x_anchor)
    
    # Compute Jacobian of F at x_t
    h = 1e-6
    J = np.zeros((2, 2))
    Fx_t = _safe_F(x_t, F)
    if Fx_t is None:
        return False, x_t, t_current
        
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_p = _safe_F(x_t + e, F)
        if Fx_p is None:
            return False, x_t, t_current
        J[:, j] = (Fx_p - Fx_t) / h
        
    try:
        J_inv = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        return False, x_t, t_current
        
    # Predictor step
    dt = t_target - t_current
    # dx/dt = - (1-t_current) * J_inv * (x_t - x_anchor)
    # Note: x_anchor is the initial x0 passed to find_root_system
    # We need to pass x_anchor to this function. Let's adjust signature or use closure/global.
    # For now, let's assume x_anchor is accessible. 
    # Actually, I'll implement a dedicated homotopy function that takes x_anchor.
    
    return False, x_t, t_current

def _trace_homotopy(F, x_anchor, max_steps=100):
    """
    Trace the homotopy path from t=1 (x=x_anchor) to t=0 (root of F).
    H(x, t) = (1-t)F(x) + t(x - x_anchor)
    """
    x = x_anchor.copy()
    t = 1.0
    
    # Initial step size
    dt = 0.1
    
    for step in range(max_steps):
        if _time_left() < 0.5:
            break
        if t <= 0.0:
            break
            
        # Target t for this step
        t_target = max(0.0, t - dt)
        
        # Predictor: 
        # We need J_F(x) and (x - x_anchor)
        h = 1e-6
        J = np.zeros((2, 2))
        Fx = _safe_F(x, F)
        if Fx is None:
            break
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_p = _safe_F(x + e, F)
            if Fx_p is None:
                break
            J[:, j] = (Fx_p - Fx) / h
            
        try:
            J_inv = np.linalg.inv(J)
        except np.linalg.LinAlgError:
            # If singular, try a smaller step or abort
            dt *= 0.5
            if dt < 1e-6:
                break
            continue
            
        # dx/dt = - (1-t) * J_inv * (x - x_anchor)
        dx_dt = -(1.0 - t) * (J_inv @ (x - x_anchor))
        
        x_pred = x + dx_dt * (t_target - t)
        
        # Corrector: Newton's method to solve H(x_new, t_target) = 0
        # H(x, t_target) = (1-t_target)F(x) + t_target(x - x_anchor)
        # Let's define G(x) = (1-t_target)F(x) + t_target(x - x_anchor)
        # We want G(x) = 0.
        # Newton step: x_new = x_pred - J_G(x_pred)^{-1} G(x_pred)
        # J_G(x) = (1-t_target) J_F(x) + t_target I
        
        x_new = x_pred.copy()
        success = False
        
        for _ in range(10): # Max 10 Newton iterations for corrector
            if _time_left() < 0.5:
                return x, t # Return current best
                
            Fx_new = _safe_F(x_new, F)
            if Fx_new is None:
                break
                
            G_x = (1.0 - t_target) * Fx_new + t_target * (x_new - x_anchor)
            norm_G = np.linalg.norm(G_x)
            
            if norm_G < 1e-12:
                success = True
                break
                
            # Compute Jacobian of G at x_new
            J_F_new = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_p_new = _safe_F(x_new + e, F)
                if Fx_p_new is None:
                    break
                J_F_new[:, j] = (Fx_p_new - Fx_new) / h
            else:
                J_G = (1.0 - t_target) * J_F_new + t_target * np.eye(2)
                try:
                    dx = np.linalg.solve(J_G, -G_x)
                    x_new = x_new + dx
                    if not np.all(np.isfinite(x_new)):
                        break
                except np.linalg.LinAlgError:
                    break
            if not success:
                break
                
        if success:
            x = x_new
            t = t_target
            # Increase step size if successful
            dt = min(dt * 1.5, 0.5)
        else:
            # Backtrack: reduce step size
            dt *= 0.5
            if dt < 1e-6:
                break
                
    return x, t

def find_root_system(F, x0):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    x0 = np.asarray(x0, dtype=float).flatten()
    if x0.shape != (2,):
        x0 = np.array([0.0, 0.0])
    
    best_x = x0.copy()
    best_norm = _safe_norm_F(best_x, F)
    
    # Strategy 1: Try Newton polish directly from x0 first (cheap if x0 is close)
    x_candidate = _newton_polish(F, x0)
    norm_candidate = _safe_norm_F(x_candidate, F)
    if norm_candidate < best_norm:
        best_x = x_candidate
        best_norm = norm_candidate
    
    if best_norm <= 1e-8:
        return best_x
        
    # Strategy 2: Homotopy Method
    # Trace from x0 (t=1) to root (t=0)
    if _time_left() > 2.0:
        x_homo, t_final = _trace_homotopy(F, x0)
        if t_final < 1e-6: # Reached t=0
            # Polish the result
            x_polished = _newton_polish(F, x_homo)
            norm_polished = _safe_norm_F(x_polished, F)
            if norm_polished < best_norm:
                best_x = x_polished
                best_norm = norm_polished
                
            if best_norm <= 1e-8:
                return best_x
        else:
            # Even if t didn't reach 0, the x_homo might be closer
            norm_homo = _safe_norm_F(x_homo, F)
            if norm_homo < best_norm:
                best_x = x_homo
                best_norm = norm_homo
                x_polished = _newton_polish(F, best_x)
                norm_polished = _safe_norm_F(x_polished, F)
                if norm_polished < best_norm:
                    best_x = x_polished
                    best_norm = norm_polished
                if best_norm <= 1e-8:
                    return best_x

    # Strategy 3: Fallback to Nelder-Mead if homotopy failed
    if _time_left() > 2.0:
        try:
            res = minimize(lambda x: np.dot(F(x), F(x)), x0, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
            x_candidate = res.x
            norm_candidate = _safe_norm_F(x_candidate, F)
            if norm_candidate < best_norm:
                best_x = x_candidate
                best_norm = norm_candidate
            
            if best_norm <= 1e-8:
                return best_x
                
            x_polished = _newton_polish(F, best_x)
            norm_polished = _safe_norm_F(x_polished, F)
            if norm_polished < best_norm:
                best_x = x_polished
                best_norm = norm_polished
                
            if best_norm <= 1e-8:
                return best_x
        except Exception:
            pass

    # Strategy 4: Random restarts with Newton polish
    if _time_left() > 1.0:
        rng = np.random.default_rng(123)
        for i in range(10):
            if _time_left() < 0.5:
                break
            offset = rng.uniform(-10.0, 10.0, size=2)
            x_init = x0 + offset
            x_candidate = _newton_polish(F, x_init, max_iter=30)
            norm_candidate = _safe_norm_F(x_candidate, F)
            if norm_candidate < best_norm:
                best_x = x_candidate
                best_norm = norm_candidate
            if best_norm <= 1e-8:
                return best_x

    return best_x
# EVOLVE-BLOCK-END
