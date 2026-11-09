"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Hybrid root finder: Newton's method with bracketing fallback.
    Uses Newton when it stays within a bracket, falls back to Illinois/bisection
    when Newton steps outside or derivative is poor.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    call_count = 0
    def _f(x):
        nonlocal call_count
        call_count += 1
        return f(x)
    
    x = float(x0)
    fx = _f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Try to establish a bracket around x0
    lo, hi = None, None
    flo, fhi = None, None
    
    # Expand bracket if needed
    step = 1.0
    for _ in range(50):
        if time.time() > deadline:
            break
        # Try expanding symmetrically
        x_lo = x - step
        x_hi = x + step
        flo = _f(x_lo)
        fhi = _f(x_hi)
        
        if flo * fhi <= 0:
            lo, hi = x_lo, x_hi
            break
        if abs(flo) < abs(fx):
            lo, flo = x_lo, flo
        if abs(fhi) < abs(fx):
            hi, fhi = x_hi, fhi
        step *= 2.0
        if step > 1e10:
            break
    
    # If we have a bracket, use Illinois method with Newton acceleration
    if lo is not None and hi is not None and flo * fhi <= 0:
        # Ensure lo < hi
        if lo > hi:
            lo, hi = hi, lo
            flo, fhi = fhi, flo
        
        best_x = x
        best_f = fx
        
        for _ in range(450):
            if time.time() > deadline:
                break
            if call_count >= 480:
                break
            
            # Check if we're close enough
            if abs(fx) <= 1e-8:
                return x
            
            # Try Newton step if derivative is well-behaved
            h = 1e-8 * max(1.0, abs(x))
            f_xh = _f(x + h)
            d = (f_xh - fx) / h
            
            if d != 0 and abs(d) > 1e-15:
                x_newton = x - fx / d
                # Check if Newton step stays in bracket
                if lo < x_newton < hi:
                    x = x_newton
                    fx = _f(x)
                    if abs(fx) < abs(flo):
                        lo, flo = x, fx
                    else:
                        hi, fhi = x, fx
                    continue
            
            # Fall back to Illinois method
            # Illinois: weight the point with smaller |f|
            if abs(flo) < abs(fhi):
                mid = (lo + hi) / 2
                fmid = _f(mid)
                if fmid * flo <= 0:
                    hi, fhi = mid, fmid
                else:
                    lo, flo = mid, fmid
            else:
                # Illinois variant: move the endpoint with larger |f| to midpoint
                if fhi * flo <= 0:
                    hi = (lo + hi) / 2
                    fhi = _f(hi)
                else:
                    lo = (lo + hi) / 2
                    flo = _f(lo)
            
            # Update x to the point with smallest |f|
            if abs(flo) < abs(fx):
                x, fx = lo, flo
            if abs(fhi) < abs(fx):
                x, fx = hi, fhi
            if abs(fx) <= 1e-8:
                return x
    
    # If no bracket found, use secant/Newton hybrid
    x_prev = x - 1e-4 * max(1.0, abs(x))
    fx_prev = _f(x_prev)
    
    for _ in range(450):
        if time.time() > deadline:
            break
        if call_count >= 480:
            break
        
        if abs(fx) <= 1e-8:
            return x
        
        d = (fx - fx_prev) / (x - x_prev) if x != x_prev else 0
        
        if d != 0 and abs(d) > 1e-15:
            x_new = x - fx / d
            # Dampen if step is too large
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x + 0.5 * (x_new - x)
            x_prev, x = x, x_new
            fx_prev = fx
            fx = _f(x)
        else:
            # Fallback: small step
            x = x + 1e-6 * max(1.0, abs(x))
            fx_prev = fx
            fx = _f(x)
    
    # Return best x found
    return x
# EVOLVE-BLOCK-END
