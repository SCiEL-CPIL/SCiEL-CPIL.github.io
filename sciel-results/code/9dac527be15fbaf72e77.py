# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, root as scipy_root

# We use a multi-start strategy with a mix of global and local methods.
# The key insight is that the 3 systems are fixed, so we can be robust by trying many starts.
# We will use a combination of:
# 1. A quick local check at x0.
# 2. A set of deterministic "smart" initial points around x0.
# 3. Nelder-Mead from a few of these points.
# 4. Newton polish from the best Nelder-Mead result.
# 5. If all else fails, a simple grid search followed by Newton.

# We track time to avoid exceeding the 120s limit (with margin).
# Each call should ideally take < 20s.
_CALL_DEADLINE = 20.0

def _safe_norm(F, x):
    """Safely compute ||F(x)||, returning inf on error."""
    try:
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            return np.inf
        return np.linalg.norm(Fx)
    except Exception:
        return np.inf

def _newton_polish(F, x, max_iter=30, tol=1e-12):
    """
    Newton's method with finite difference Jacobian.
    Returns (x, norm_F, success).
    """
    x = np.array(x, dtype=float)
    for _ in range(max_iter):
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return x, np.inf, False
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx <= tol:
                return x, norm_Fx, True
            
            # Compute Jacobian using central differences for better accuracy
            h = 1e-7
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_plus = F(x + e)
                Fx_minus = F(x - e)
                if not (np.all(np.isfinite(Fx_plus)) and np.all(np.isfinite(Fx_minus))):
                    # Fall back to forward difference if central fails
                    Fx_plus = F(x + e)
                    if not np.all(np.isfinite(Fx_plus)):
                        return x, norm_Fx, False
                    J[:, j] = (Fx_plus - Fx) / h
                else:
                    J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
            
            dx = np.linalg.solve(J, Fx)
            x_new = x - dx
            if not np.all(np.isfinite(x_new)):
                return x, norm_Fx, False
            x = x_new
        except np.linalg.LinAlgError:
            return x, np.inf, False
        except Exception:
            return x, np.inf, False
    
    try:
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            return x, np.inf, False
        norm_Fx = np.linalg.norm(Fx)
        return x, norm_Fx, norm_Fx <= tol
    except Exception:
        return x, np.inf, False

def _nelder_mead(F, x0, max_iter=200):
    """
    Run Nelder-Mead on ||F(x)||^2.
    Returns (x, norm_F).
    """
    def objective(x):
        try:
            fx = F(x)
            if not np.all(np.isfinite(fx)):
                return 1e100
            return np.dot(fx, fx)
        except Exception:
            return 1e100

    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-10, 'fatol': 1e-20, 'maxiter': max_iter, 'maxfev': 300})
        x_opt = res.x
        norm_opt = _safe_norm(F, x_opt)
        return x_opt, norm_opt
    except Exception:
        return x0, _safe_norm(F, x0)

def find_root_system(F, x0):
    """
    Finds a root x such that ||F(x)|| <= 1e-8.
    """
    start_time = time.time()
    x0 = np.array(x0, dtype=float)
    
    # 1. Check if x0 is already a root
    norm0 = _safe_norm(F, x0)
    if norm0 <= 1e-8:
        return x0
    
    best_x = x0
    best_norm = norm0
    
    # 2. Generate a set of initial points
    # We'll try points at various distances and angles around x0
    # This is a deterministic set to ensure reproducibility and coverage.
    initial_points = [x0]
    
    # Add points in a circle around x0
    for r in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
            dx = r * np.cos(theta)
            dy = r * np.sin(theta)
            x_try = x0 + np.array([dx, dy])
            initial_points.append(x_try)
    
    # 3. Try Newton from a few of the best initial points
    # First, do a quick screening of the initial points
    screened_points = []
    for x_try in initial_points:
        if time.time() - start_time > _CALL_DEADLINE * 0.3:
            break
        norm_try = _safe_norm(F, x_try)
        screened_points.append((norm_try, x_try))
        if norm_try < best_norm:
            best_norm = norm_try
            best_x = x_try
        if norm_try <= 1e-8:
            return x_try
    
    # Sort by norm
    screened_points.sort(key=lambda t: t[0])
    
    # Try Newton from the top 5 best points
    for norm_try, x_try in screened_points[:5]:
        if time.time() - start_time > _CALL_DEADLINE * 0.7:
            break
        x_new, norm_new, success = _newton_polish(F, x_try, max_iter=30, tol=1e-12)
        if success and norm_new <= 1e-8:
            return x_new
        if norm_new < best_norm:
            best_norm = norm_new
            best_x = x_new
    
    # 4. If Newton didn't work, try Nelder-Mead from the best point
    if time.time() - start_time < _CALL_DEADLINE * 0.8:
        x_nm, norm_nm = _nelder_mead(F, best_x, max_iter=200)
        if norm_nm < best_norm:
            best_norm = norm_nm
            best_x = x_nm
        if norm_nm <= 1e-8:
            return best_x
        
        # Try Newton polish from the Nelder-Mead result
        x_new, norm_new, success = _newton_polish(F, best_x, max_iter=30, tol=1e-12)
        if success and norm_new <= 1e-8:
            return x_new
        if norm_new < best_norm:
            best_norm = norm_new
            best_x = x_new
    
    # 5. If all else fails, return the best point we found
    return best_x
# EVOLVE-BLOCK-END
