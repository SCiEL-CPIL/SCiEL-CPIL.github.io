"""Scalar root finder for 3 fixed smooth polynomial equations."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses a multi-strategy approach with deadline guard.
    """
    DEADLINE = 140.0  # Safety margin before 150s kill
    start_time = time.time()
    
    def check_deadline():
        return (time.time() - start_time) < DEADLINE
    
    def is_finite(x):
        return math.isfinite(x)
    
    def safe_f(x):
        try:
            val = f(x)
            if not is_finite(val):
                return None
            return float(val)
        except:
            return None
    
    # Track best solution found
    best_x = None
    best_fx = float('inf')
    
    def update_best(x, fx):
        nonlocal best_x, best_fx
        if is_finite(x) and is_finite(fx) and abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
        if best_x is not None and abs(best_fx) <= 1e-8:
            return True
        return False
    
    # First try at x0
    x0 = float(x0)
    fx0 = safe_f(x0)
    if fx0 is not None:
        update_best(x0, fx0)
        if abs(fx0) <= 1e-8:
            return x0
    
    # Strategy 1: Secant method (1 call per step, faster convergence for smooth polynomials)
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        x1 = x0
        # Choose x2 with a small offset to avoid division by zero issues
        offset = 1e-4 * max(1.0, abs(x0))
        x2 = x0 + offset
        fx1 = safe_f(x1)
        fx2 = safe_f(x2)
        
        if fx1 is not None and fx2 is not None:
            # If fx1 is already good, return it
            if abs(fx1) <= 1e-8:
                return x1
            # If fx2 is already good, return it
            if abs(fx2) <= 1e-8:
                return x2
            
            for _ in range(200):
                if not check_deadline():
                    break
                if abs(fx2) <= 1e-8:
                    return x2
                update_best(x2, fx2)
                if abs(fx1) <= 1e-8:
                    return x1
                update_best(x1, fx1)
                
                denom = fx2 - fx1
                if denom == 0 or not is_finite(denom):
                    break
                x_new = x2 - fx2 * (x2 - x1) / denom
                if not is_finite(x_new):
                    break
                # Keep points distinct
                if abs(x_new - x1) < 1e-14:
                    x_new = x1 + 1e-10
                if abs(x_new - x2) < 1e-14:
                    x_new = x2 + 1e-10
                
                x1, fx1 = x2, fx2
                x2 = x_new
                fx2 = safe_f(x2)
                if fx2 is None:
                    break
    
    # Strategy 2: Bisection with bracketing search (fallback if secant fails)
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        # Try to find a bracket around x0
        a, b = None, None
        fa, fb = None, None
        
        # Expand search range
        scale = 1.0
        for _ in range(50):
            if not check_deadline():
                break
            a_try = x0 - scale
            b_try = x0 + scale
            
            fa_try = safe_f(a_try)
            fb_try = safe_f(b_try)
            
            if fa_try is not None and fb_try is not None:
                if fa_try * fb_try <= 0:
                    a, b, fa, fb = a_try, b_try, fa_try, fb_try
                    break
                # Check if either endpoint is good enough
                if abs(fa_try) <= 1e-8:
                    return a_try
                if abs(fb_try) <= 1e-8:
                    return b_try
            
            scale *= 2
            if scale > 1e6:
                break
        
        if a is not None and b is not None:
            for _ in range(200):
                if not check_deadline():
                    break
                mid = (a + b) / 2.0
                fmid = safe_f(mid)
                if fmid is None:
                    break
                if abs(fmid) <= 1e-8:
                    return mid
                update_best(mid, fmid)
                
                if fa * fmid <= 0:
                    b = mid
                    fb = fmid
                else:
                    a = mid
                    fa = fmid
    
    # Strategy 3: Global search with multiple starting points (last resort)
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        # Try several starting points
        offsets = [0.01, -0.01, 0.1, -0.1, 1.0, -1.0, 10.0, -10.0,
                   0.001, -0.001, 100.0, -100.0]
        for offset in offsets:
            if not check_deadline():
                break
            x_start = x0 + offset * max(1.0, abs(x0))
            fx_start = safe_f(x_start)
            if fx_start is None:
                continue
            if abs(fx_start) <= 1e-8:
                return x_start
            update_best(x_start, fx_start)
            
            # Quick secant from this point
            x1 = x_start
            x2 = x_start + 1e-4 * max(1.0, abs(x_start))
            fx1 = fx_start
            fx2 = safe_f(x2)
            if fx2 is None:
                continue
            if abs(fx2) <= 1e-8:
                return x2
            
            for _ in range(30):
                if not check_deadline():
                    break
                if abs(fx2) <= 1e-8:
                    return x2
                denom = fx2 - fx1
                if denom == 0 or not is_finite(denom):
                    break
                x_new = x2 - fx2 * (x2 - x1) / denom
                if not is_finite(x_new):
                    break
                x1, fx1 = x2, fx2
                x2 = x_new
                fx2 = safe_f(x2)
                if fx2 is None:
                    break
    
    # Return best found
    if best_x is not None:
        return best_x
    return x0
# EVOLVE-BLOCK-END
