"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust scalar root finder for smooth polynomial equations.
    Uses a hybrid approach: initial bracketing, then bisection + secant/Newton
    with safety checks and a time guard.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    TOL = 1e-10  # internal tolerance, stricter than required 1e-8
    MAX_CALLS = 490  # stay under 500 per equation
    
    x = float(x0)
    fx = None
    call_count = 0
    
    def _check_time():
        return time.time() < deadline
    
    def _safe_f(x_val):
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return None
        try:
            val = f(x_val)
            call_count += 1
            if not math.isfinite(val):
                return None
            return val
        except Exception:
            call_count += 1
            return None
    
    # Phase 1: Try to find a bracket around x0
    # Expand outward to find sign change
    a, b = x, x
    fa, fb = None, None
    step = 1.0
    
    # First, evaluate at x0
    fx = _safe_f(x)
    if fx is None:
        return x
    if abs(fx) <= TOL:
        return x
    
    # Try to find a bracket by expanding
    for _ in range(30):
        if not _check_time():
            break
        
        # Try right expansion
        xr = x + step
        fr = _safe_f(xr)
        if fr is not None:
            if abs(fr) <= TOL:
                return xr
            if fx is not None and fx * fr < 0:
                a, b = x, xr
                fa, fb = fx, fr
                break
        
        # Try left expansion
        xl = x - step
        fl = _safe_f(xl)
        if fl is not None:
            if abs(fl) <= TOL:
                return xl
            if fx is not None and fx * fl < 0:
                a, b = xl, x
                fa, fb = fl, fx
                break
        
        step *= 2.0
        if step > 1e6:
            break
    
    # If we found a bracket, use hybrid bisection + secant
    if a is not None and b is not None and fa is not None and fb is not None:
        # Ensure fa and fb have opposite signs
        if fa * fb > 0:
            a, b, fa, fb = b, a, fb, fa
        
        x_lo, x_hi = a, b
        f_lo, f_hi = fa, fb
        x_best = (x_lo + x_hi) / 2.0
        f_best = None
        
        # Hybrid method: use secant when safe, bisection otherwise
        x_prev = x_hi
        f_prev = f_hi
        x_curr = x_lo
        f_curr = f_lo
        
        for _ in range(60):
            if not _check_time():
                break
            if call_count >= MAX_CALLS - 5:
                break
            
            # Check if either endpoint is good enough
            if abs(f_lo) <= TOL:
                return x_lo
            if abs(f_hi) <= TOL:
                return x_hi
            
            # Midpoint
            x_mid = (x_lo + x_hi) / 2.0
            f_mid = _safe_f(x_mid)
            if f_mid is None:
                break
            if abs(f_mid) <= TOL:
                return x_mid
            
            # Try secant step
            if f_prev is not None and f_curr is not None and f_prev != f_curr:
                x_sec = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
                
                # Check if secant point is within bracket
                if x_lo < x_sec < x_hi:
                    f_sec = _safe_f(x_sec)
                    if f_sec is not None:
                        if abs(f_sec) <= TOL:
                            return x_sec
                        # Update bracket based on sign
                        if f_sec * f_lo < 0:
                            x_hi = x_sec
                            f_hi = f_sec
                            x_prev, f_prev = x_curr, f_curr
                            x_curr, f_curr = x_sec, f_sec
                        else:
                            x_lo = x_sec
                            f_lo = f_sec
                            x_prev, f_prev = x_curr, f_curr
                            x_curr, f_curr = x_sec, f_sec
                        continue
            
            # Fallback: bisection
            if f_mid * f_lo < 0:
                x_hi = x_mid
                f_hi = f_mid
            else:
                x_lo = x_mid
                f_lo = f_mid
            
            # Update for secant
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_mid, f_mid
        
        # Return best estimate
        if f_best is not None and abs(f_best) <= TOL:
            return x_best
        # Return midpoint or whichever endpoint has smaller |f|
        if abs(f_lo) < abs(f_hi):
            return x_lo
        else:
            return x_hi
    
    # Phase 2: No bracket found, use Newton's method with fallback
    # Try Newton with finite difference derivative
    x = float(x0)
    for _ in range(40):
        if not _check_time():
            break
        if call_count >= MAX_CALLS - 3:
            break
        
        fx = _safe_f(x)
        if fx is None:
            break
        if abs(fx) <= TOL:
            return x
        
        h = 1e-7 * max(1.0, abs(x))
        fx_plus = _safe_f(x + h)
        if fx_plus is None:
            break
        
        d = (fx_plus - fx) / h
        if d == 0 or not math.isfinite(d):
            # Try smaller step
            h = 1e-9 * max(1.0, abs(x))
            fx_plus = _safe_f(x + h)
            if fx_plus is None:
                break
            d = (fx_plus - fx) / h
            if d == 0 or not math.isfinite(d):
                break
        
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        
        # Damping: limit step size to avoid overshooting
        max_step = 1.0 * max(1.0, abs(x))
        if abs(x_new - x) > max_step:
            x_new = x + max_step * (1 if x_new > x else -1)
        
        x = x_new
    
    # Final check
    fx_final = _safe_f(x)
    if fx_final is not None and abs(fx_final) <= TOL:
        return x
    
    return x
# EVOLVE-BLOCK-END
