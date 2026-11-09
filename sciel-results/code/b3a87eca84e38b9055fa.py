import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 145.0
    max_calls = 480
    call_count = 0
    
    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    def f_call(x):
        nonlocal call_count
        call_count += 1
        return f(x)

    x = float(x0)
    fx = f_call(x)
    
    if not math.isfinite(fx):
        return x
    
    if abs(fx) <= 1e-8:
        return x

    # Try to find a bracket with expanding window
    a = x
    fa = fx
    h = 1.0
    b = None
    fb = None
    
    for _ in range(20):
        if not time_ok():
            break
        b = a + h
        fb = f_call(b)
        if not math.isfinite(fb):
            h *= 2
            continue
        if fa * fb < 0:
            break
        a = b
        fa = fb
        h *= 2
        
    if b is not None and fa * fb < 0:
        # Illinois algorithm for fast convergence
        # Illinois: alternate between secant and bisection, but when using secant,
        # shrink the opposite end to maintain bracket while accelerating convergence
        
        # Initialize with midpoint
        c = 0.5 * (a + b)
        fc = f_call(c)
        
        if abs(fc) <= 1e-8:
            return c
        
        # Set up Illinois: a, b are bracket endpoints, c is current best estimate
        # We'll maintain: lo, hi (bracket), and use Illinois stepping
        lo = a
        hi = b
        flo = fa
        fhi = fb
        
        # Determine which side c is on
        if flo * fc < 0:
            hi = c
            fhi = fc
        else:
            lo = c
            flo = fc
            
        # Illinois iteration
        for _ in range(120):
            if not time_ok():
                break
                
            # Check convergence
            if abs(flo) <= 1e-8:
                return lo
            if abs(fhi) <= 1e-8:
                return hi
                
            # Interval width check
            if abs(hi - lo) < 1e-14 * max(1.0, abs(lo), abs(hi)):
                return 0.5 * (lo + hi)
                
            # Illinois step:
            # If |fhi| < |flo|, the root is closer to hi, so we shrink lo
            # Otherwise, shrink hi
            if abs(fhi) < abs(flo):
                # Root closer to hi, use secant from (lo, hi) but shrink lo
                # Secant point: x_new = lo - flo*(lo-hi)/(flo-fhi)
                if fhi != flo:
                    x_new = lo - flo * (lo - hi) / (flo - fhi)
                else:
                    x_new = 0.5 * (lo + hi)
                
                # If x_new is outside [lo, hi], use bisection
                if x_new < lo or x_new > hi:
                    x_new = 0.5 * (lo + hi)
                
                fx_new = f_call(x_new)
                
                if not math.isfinite(fx_new):
                    x_new = 0.5 * (lo + hi)
                    fx_new = f_call(x_new)
                    
                if abs(fx_new) <= 1e-8:
                    return x_new
                    
                # Update bracket
                if flo * fx_new < 0:
                    hi = x_new
                    fhi = fx_new
                    # Illinois: shrink lo to x_new (the old lo becomes the new "shrunken" end)
                    # Actually in Illinois, we replace the farther endpoint
                    lo = x_new
                    flo = fx_new
                else:
                    lo = x_new
                    flo = fx_new
                    # Shrink: the old lo is replaced, but we keep hi
                    # Illinois rule: when we update lo, we shrink the interval by moving lo
                    # But we also need to maintain the property that we alternate
                    # Standard Illinois: if we move lo, set new lo = x_new, and the "old" lo value is discarded
                    # The key is: we always shrink the end that had the smaller |f|
                    # So if |fhi| < |flo|, we're shrinking lo, so new lo = x_new, and we don't update hi
                    # But we need to track which end was shrunk for the next iteration
                    # Let's use a different approach: track the last two points
                    
            else:
                # Root closer to lo, use secant from (lo, hi) but shrink hi
                if fhi != flo:
                    x_new = hi - fhi * (hi - lo) / (fhi - flo)
                else:
                    x_new = 0.5 * (lo + hi)
                    
                if x_new < lo or x_new > hi:
                    x_new = 0.5 * (lo + hi)
                    
                fx_new = f_call(x_new)
                
                if not math.isfinite(fx_new):
                    x_new = 0.5 * (lo + hi)
                    fx_new = f_call(x_new)
                    
                if abs(fx_new) <= 1e-8:
                    return x_new
                    
                if fhi * fx_new < 0:
                    lo = x_new
                    flo = fx_new
                    hi = x_new
                    fhi = fx_new
                else:
                    hi = x_new
                    fhi = fx_new
                    
        # Return the best point
        if abs(flo) <= abs(fhi):
            return lo
        else:
            return hi

    else:
        # No bracket found - try Newton with damping and fallback
        x = float(x0)
        fx = f_call(x)
        
        for _ in range(80):
            if not time_ok():
                break
            if abs(fx) <= 1e-8:
                return x
                
            # Finite difference derivative
            h = 1e-7 * max(1.0, abs(x))
            fx_plus = f_call(x + h)
            d = (fx_plus - fx) / h
            
            if d == 0 or not math.isfinite(d):
                # Try larger h
                h = 1e-5 * max(1.0, abs(x))
                fx_plus = f_call(x + h)
                d = (fx_plus - fx) / h
                if d == 0 or not math.isfinite(d):
                    break
                    
            x_new = x - fx / d
            
            if not math.isfinite(x_new):
                break
                
            # Damping
            step = x_new - x
            if abs(step) > 5.0:
                x_new = x + 5.0 * math.copysign(1.0, step)
                
            x = x_new
            fx = f_call(x)
            
        return x
