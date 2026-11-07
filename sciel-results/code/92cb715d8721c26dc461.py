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
    TOL = 1e-9  # internal tolerance, slightly stricter than required 1e-8
    MAX_CALLS = 495  # stay under 500 per equation
    
    x = float(x0)
    call_count = 0
    best_x = x
    best_f = float('inf')
    
    def _check_time():
        return time.time() < deadline
    
    def _safe_f(x_val):
        nonlocal call_count, best_x, best_f
        if call_count >= MAX_CALLS:
            return None
        try:
            val = f(x_val)
            call_count += 1
            if not math.isfinite(val):
                return None
            if abs(val) < abs(best_f):
                best_f = val
                best_x = x_val
            return val
        except Exception:
            call_count += 1
            return None
    
    # Phase 1: Try to find a bracket around x0
    fx = _safe_f(x)
    if fx is None:
        return best_x
    if abs(fx) <= TOL:
        return x
    
    # Try multiple offsets before expanding (saves calls for nearby roots)
    offsets = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0,
               256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0]
    a, b = None, None
    fa, fb = None, None
    
    for step in offsets:
        if not _check_time() or call_count >= MAX_CALLS - 10:
            break
        
        # Try right
        xr = x + step
        fr = _safe_f(xr)
        if fr is not None:
            if abs(fr) <= TOL:
                return xr
            if fx * fr < 0:
                a, b = x, xr
                fa, fb = fx, fr
                break
        
        # Try left
        xl = x - step
        fl = _safe_f(xl)
        if fl is not None:
            if abs(fl) <= TOL:
                return xl
            if fx * fl < 0:
                a, b = xl, x
                fa, fb = fl, fx
                break
    
    # If we found a bracket, use hybrid bisection + secant
    if a is not None and b is not None and fa is not None and fb is not None:
        if fa * fb > 0:
            a, b, fa, fb = b, a, fb, fa
        
        x_lo, x_hi = a, b
        f_lo, f_hi = fa, fb
        
        # Keep two most recent points for secant
        x_prev, f_prev = x_hi, f_hi
        x_curr, f_curr = x_lo, f_lo
        
        for _ in range(80):
            if not _check_time():
                break
            if call_count >= MAX_CALLS - 3:
                break
            
            if abs(f_lo) <= TOL:
                return x_lo
            if abs(f_hi) <= TOL:
                return x_hi
            
            # Try secant first if we have two distinct points
            used_secant = False
            if x_prev != x_curr and f_prev != f_curr:
                x_sec = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
                # Check if secant point is within bracket (with small margin)
                if x_lo + 1e-15 < x_sec < x_hi - 1e-15:
                    f_sec = _safe_f(x_sec)
                    if f_sec is not None:
                        if abs(f_sec) <= TOL:
                            return x_sec
                        # Update bracket based on sign
                        if f_sec * f_lo < 0:
                            x_hi = x_sec
                            f_hi = f_sec
                        else:
                            x_lo = x_sec
                            f_lo = f_sec
                        # Update secant history
                        x_prev, f_prev = x_curr, f_curr
                        x_curr, f_curr = x_sec, f_sec
                        used_secant = True
            
            if not used_secant:
                # Bisection
                x_mid = (x_lo + x_hi) / 2.0
                f_mid = _safe_f(x_mid)
                if f_mid is None:
                    break
                if abs(f_mid) <= TOL:
                    return x_mid
                
                if f_mid * f_lo < 0:
                    x_hi = x_mid
                    f_hi = f_mid
                else:
                    x_lo = x_mid
                    f_lo = f_mid
                
                # Update secant history
                x_prev, f_prev = x_curr, f_curr
                x_curr, f_curr = x_mid, f_mid
        
        # Return whichever endpoint has smaller |f|
        if abs(f_lo) <= abs(f_hi):
            return x_lo
        else:
            return x_hi
    
    # Phase 2: No bracket found, use secant method with damping
    x_curr = x
    f_curr = fx
    h = 0.1 * max(1.0, abs(x))
    x_prev = x + h
    f_prev = _safe_f(x_prev)
    
    if f_prev is None:
        return best_x
    
    for _ in range(40):
        if not _check_time():
            break
        if call_count >= MAX_CALLS - 3:
            break
        
        if abs(f_curr) <= TOL:
            return x_curr
        
        # Secant step
        if f_prev != f_curr and x_prev != x_curr:
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        else:
            # Fallback: small step
            x_new = x_curr + h
            h *= 0.5
        
        if not math.isfinite(x_new):
            break
        
        # Damping: limit step size
        max_step = 2.0 * max(1.0, abs(x_curr))
        if abs(x_new - x_curr) > max_step:
            x_new = x_curr + max_step * (1 if x_new > x_curr else -1)
        
        f_new = _safe_f(x_new)
        if f_new is None:
            break
        
        if abs(f_new) <= TOL:
            return x_new
        
        # Update history: keep the two points with smallest |f|
        if abs(f_new) < abs(f_curr):
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_new, f_new
        else:
            x_prev, f_prev = x_curr, f_curr
    
    return best_x
# EVOLVE-BLOCK-END
