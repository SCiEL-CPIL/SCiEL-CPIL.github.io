"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize_scalar, brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 for oscillatory equations.
    
    Strategy:
    1. Check x0.
    2. Local scan for sign change brackets around x0 with expanding intervals.
    3. If no bracket, use minimize_scalar on f(x)^2 to find closest approach.
    4. If minimization finds a good point, try to bracket around it.
    5. Polish with Brent's method if a bracket is found.
    """
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    x = float(x0)
    best_x = x
    best_f = abs(f(x))
    
    if best_f <= 1e-8:
        return x
    
    # Phase 1: Local scan for sign change
    # Try expanding intervals around x0
    # We evaluate f(x0-d) and f(x0+d)
    # If sign change, solve with brentq
    
    deltas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]
    
    for d in deltas:
        if time_left() < 5.0:
            break
            
        a = x0 - d
        b = x0 + d
        
        try:
            fa = f(a)
            fb = f(b)
        except Exception:
            continue
            
        if not np.isfinite(fa) or not np.isfinite(fb):
            continue
            
        # Update bests
        if abs(fa) < best_f:
            best_f = abs(fa)
            best_x = a
        if abs(fb) < best_f:
            best_f = abs(fb)
            best_x = b
            
        if best_f <= 1e-8:
            return best_x
            
        # Check for sign change
        if fa * fb < 0:
            try:
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-14, maxiter=50)
                if np.isfinite(root):
                    froot = f(root)
                    if np.isfinite(froot) and abs(froot) <= 1e-8:
                        return root
                    if abs(froot) < best_f:
                        best_f = abs(froot)
                        best_x = root
            except Exception:
                pass
    
    # Phase 2: Global minimization if no bracket found
    # Use minimize_scalar on f(x)^2
    # We use the largest scale from Phase 1 as the bound, or a default if Phase 1 was skipped/failed
    max_scale = 1e4
    if len(deltas) > 0:
        max_scale = deltas[-1]
        
    def objective(x):
        fx = f(x)
        if not np.isfinite(fx):
            return 1e10
        return fx * fx

    lo = x0 - max_scale
    hi = x0 + max_scale
    
    try:
        result = minimize_scalar(
            objective,
            bounds=(lo, hi),
            method='bounded',
            options={'xatol': 1e-10, 'maxiter': 100}
        )
    except Exception:
        result = None
        
    if result and result.success and np.isfinite(result.x):
        fx_at_min = f(result.x)
        if np.isfinite(fx_at_min):
            if abs(fx_at_min) < best_f:
                best_f = abs(fx_at_min)
                best_x = result.x
            if best_f <= 1e-8:
                return best_x
            
            # Phase 3: Try to bracket around the minimizer
            if best_f < 1e-4:
                for delta in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
                    if time_left() < 5.0:
                        break
                    a = best_x - delta
                    b = best_x + delta
                    try:
                        fa = f(a)
                        fb = f(b)
                        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                            root = brentq(f, a, b, xtol=1e-12, rtol=1e-14, maxiter=50)
                            if np.isfinite(root):
                                froot = f(root)
                                if np.isfinite(froot) and abs(froot) <= 1e-8:
                                    return root
                                if abs(froot) < best_f:
                                    best_f = abs(froot)
                                    best_x = root
                            break # No need to try larger deltas if we tried brentq on the smallest successful bracket
                    except Exception:
                        continue
                        
    return best_x
# EVOLVE-BLOCK-END
