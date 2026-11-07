"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find a root of f near x0 using a hybrid strategy:
    1. Try Newton's method with finite-difference derivative (fast, few calls)
    2. If Newton fails or takes too many calls, fall back to Brent's method
       with automatic bracketing.
    
    Returns x such that |f(x)| <= 1e-8, or the best x found if time is up.
    """
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    x = float(x0)
    best_x = x
    best_fx = None
    try:
        best_fx = f(x)
        if abs(best_fx) <= 1e-8:
            return x
    except Exception:
        pass
    
    # Phase 1: Newton's method (fast for well-behaved roots)
    # Use forward difference to minimize calls
    max_newton_iters = 15
    for i in range(max_newton_iters):
        if time.time() - start_time > DEADLINE:
            break
        
        try:
            fx = f(x)
        except Exception:
            break
        
        if not np.isfinite(fx):
            break
            
        if abs(fx) <= 1e-8:
            return x
        
        # Track best so far
        if best_fx is None or abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
        
        # Finite difference derivative
        h = 1e-6 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
        except Exception:
            break
        
        if not np.isfinite(fx_plus):
            break
        
        d = (fx_plus - fx) / h
        if d == 0 or not np.isfinite(d):
            break
        
        x_new = x - fx / d
        
        # Damping if step is too large
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
            x_new = x + 0.5 * np.sign(x_new - x) * max(1.0, abs(x))
        
        if not np.isfinite(x_new):
            break
            
        x = x_new
    
    # Phase 2: If Newton didn't find a root, try Brent with bracketing
    if time.time() - start_time > DEADLINE:
        return best_x
    
    # Check if we already have a good answer
    try:
        fx_check = f(x)
        if abs(fx_check) <= 1e-8:
            return x
        if best_fx is None or abs(fx_check) < abs(best_fx):
            best_x = x
            best_fx = fx_check
    except Exception:
        pass
    
    # Try to bracket the root around x0 and current x
    # Expand search interval
    a, b = None, None
    fa = None
    fb = None
    
    # Try several bracketing strategies
    for scale in [1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
        if time.time() - start_time > DEADLINE:
            break
        
        # Try bracket centered at x0
        lo = x0 - scale
        hi = x0 + scale
        
        try:
            f_lo = f(lo)
            if np.isfinite(f_lo) and abs(f_lo) <= 1e-8:
                return lo
            f_hi = f(hi)
            if np.isfinite(f_hi) and abs(f_hi) <= 1e-8:
                return hi
            
            if np.isfinite(f_lo) and np.isfinite(f_hi):
                if f_lo * f_hi < 0:
                    a, b = lo, hi
                    fa, fb = f_lo, f_hi
                    break
        except Exception:
            continue
        
        # Also try bracket centered at current x
        if abs(x - x0) > 1e-10:
            lo2 = x - scale
            hi2 = x + scale
            try:
                f_lo2 = f(lo2)
                if np.isfinite(f_lo2) and abs(f_lo2) <= 1e-8:
                    return lo2
                f_hi2 = f(hi2)
                if np.isfinite(f_hi2) and abs(f_hi2) <= 1e-8:
                    return hi2
                
                if np.isfinite(f_lo2) and np.isfinite(f_hi2):
                    if f_lo2 * f_hi2 < 0:
                        a, b = lo2, hi2
                        fa, fb = f_lo2, f_hi2
                        break
            except Exception:
                continue
    
    # If we have a bracket, use Brent's method
    if a is not None and b is not None and time.time() - start_time < DEADLINE:
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if np.isfinite(root):
                f_root = f(root)
                if abs(f_root) <= 1e-8:
                    return root
                if best_fx is None or abs(f_root) < abs(best_fx):
                    best_x = root
                    best_fx = f_root
        except Exception:
            pass
    
    # Phase 3: If no bracket found, try a more systematic scan
    if time.time() - start_time > DEADLINE:
        return best_x
    
    # Scan for sign changes in a wider range
    scan_range = max(100.0, abs(x0) * 2)
    n_steps = 200
    step_size = 2 * scan_range / n_steps
    
    prev_x = x0 - scan_range
    try:
        prev_fx = f(prev_x)
        if np.isfinite(prev_fx) and abs(prev_fx) <= 1e-8:
            return prev_x
    except Exception:
        prev_fx = None
    
    for i in range(1, n_steps + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        cur_x = x0 - scan_range + i * step_size
        try:
            cur_fx = f(cur_x)
        except Exception:
            continue
        
        if not np.isfinite(cur_fx):
            continue
            
        if abs(cur_fx) <= 1e-8:
            return cur_x
        
        if best_fx is None or abs(cur_fx) < abs(best_fx):
            best_x = cur_x
            best_fx = cur_fx
        
        if prev_fx is not None and np.isfinite(prev_fx):
            if prev_fx * cur_fx < 0:
                # Found a bracket, use Brent
                try:
                    root = brentq(f, prev_x, cur_x, xtol=1e-12, rtol=1e-12, maxiter=100)
                    if np.isfinite(root):
                        f_root = f(root)
                        if abs(f_root) <= 1e-8:
                            return root
                        if best_fx is None or abs(f_root) < abs(best_fx):
                            best_x = root
                            best_fx = f_root
                except Exception:
                    pass
        
        prev_x = cur_x
        prev_fx = cur_fx
    
    return best_x

# EVOLVE-BLOCK-END
