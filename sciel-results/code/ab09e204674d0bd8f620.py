# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Find a root of f near x0 using adaptive spectral-aware bracketing.
    1. Quick local check and Newton steps.
    2. Geometric expansion from a very small initial step to find the closest sign change.
    3. Bisection/Newton hybrid to converge.
    """
    x0 = float(x0)
    calls = 0
    
    def _f(x):
        nonlocal calls
        calls += 1
        val = f(x)
        if not math.isfinite(val):
            return float('nan')
        return float(val)
    
    # Step 1: Check x0
    fx0 = _f(x0)
    if math.isfinite(fx0) and abs(fx0) <= 1e-8:
        return x0

    # Step 2: Try Newton's method (few steps)
    x = x0
    fx = fx0
    for _ in range(5):
        if not math.isfinite(fx):
            break
        if abs(fx) <= 1e-8:
            return x
        h = 1e-7 * max(1.0, abs(x))
        fph = _f(x + h)
        if not math.isfinite(fph):
            break
        d = (fph - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        x = x_new
        fx = _f(x)
    if math.isfinite(fx) and abs(fx) <= 1e-8:
        return x

    # Step 3: Adaptive Bracketing
    # Start with a very small step to catch high-frequency roots close to x0
    # Expand geometrically until a bracket is found.
    
    # Scale based on x0 to handle different magnitudes
    scale = max(1.0, abs(x0))
    initial_step = 1e-6 * scale
    
    left_x = x0
    right_x = x0
    left_f = fx0
    right_f = fx0
    
    bracket_found = False
    a = b = 0.0
    fa = fb = 0.0
    
    # We will expand the search range [x0 - R, x0 + R]
    # We check endpoints and potentially midpoints if we suspect oscillation?
    # For now, just check endpoints of the expanding interval.
    # To ensure we don't miss a root in the "gap" when expanding, 
    # we rely on the fact that we start very small and expand by factor 2.
    # The "gap" is the annulus between previous R and new R.
    # If we only check endpoints, we might miss a root that is not at the endpoint.
    # Example: f(x) = sin(100x). Root at 0.01.
    # R=0.001: f(0.001) ~ 0.1. No sign change with f(0)=0.
    # R=0.002: f(0.002) ~ 0.2.
    # ...
    # R=0.01: f(0.01) ~ 0.
    # This works if the root is exactly at the endpoint or if the sign changes between previous endpoint and new endpoint.
    # If the root is *inside* the new ring but not at the endpoint, and the sign at the new endpoint is the same as the sign at the previous endpoint, we miss it.
    # Example: f(x) = sin(x). x0=0.
    # R=1: f(1)>0.
    # R=2: f(2)>0.
    # R=3: f(3)<0. Bracket [2,3] or [0,3]? 
    # If we track the last point with a specific sign, we can bracket.
    
    # Better: Keep track of the last point on left and right that had a sign opposite to f(x0) or just track the last evaluated point.
    
    last_left_x = x0
    last_left_f = fx0
    last_right_x = x0
    last_right_f = fx0
    
    R = initial_step
    for i in range(40):
        # Expand to new R
        new_left_x = x0 - R
        new_right_x = x0 + R
        
        new_left_f = _f(new_left_x)
        new_right_f = _f(new_right_x)
        
        # Check if new_left_f is a root
        if math.isfinite(new_left_f) and abs(new_left_f) <= 1e-8:
            return new_left_x
        # Check if new_right_f is a root
        if math.isfinite(new_right_f) and abs(new_right_f) <= 1e-8:
            return new_right_x
            
        # Check for sign change on the left side
        if math.isfinite(new_left_f) and math.isfinite(last_left_f):
            if new_left_f * last_left_f < 0:
                a = new_left_x
                b = last_left_x
                fa = new_left_f
                fb = last_left_f
                bracket_found = True
                break
        
        # Check for sign change on the right side
        if math.isfinite(new_right_f) and math.isfinite(last_right_f):
            if new_right_f * last_right_f < 0:
                a = last_right_x
                b = new_right_x
                fa = last_right_f
                fb = new_right_f
                bracket_found = True
                break
        
        # Update last points
        last_left_x = new_left_x
        last_left_f = new_left_f
        last_right_x = new_right_x
        last_right_f = new_right_f
        
        # Expand R
        R *= 2.0
        
        # Safety check: if R becomes too large, break
        if R > 1e6 * scale:
            break

    if not bracket_found:
        # If no bracket found, return best guess or x0
        # Try a final Newton step from x0
        x = x0
        fx = _f(x)
        for _ in range(50):
            if not math.isfinite(fx):
                break
            if abs(fx) <= 1e-8:
                return x
            h = 1e-8 * max(1.0, abs(x))
            fph = _f(x + h)
            if not math.isfinite(fph):
                break
            d = (fph - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            x = x_new
            fx = _f(x)
        return x if math.isfinite(x) else x0

    # Step 4: Converge using Bisection + Newton
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
    
    lo, hi = a, b
    flo, fhi = fa, fb
    
    for _ in range(100):
        if not math.isfinite(flo) or not math.isfinite(fhi):
            break
        if abs(flo) <= 1e-8:
            return lo
        if abs(fhi) <= 1e-8:
            return hi
        
        mid = (lo + hi) / 2.0
        fmid = _f(mid)
        
        if not math.isfinite(fmid):
            # Fall back to bisection logic using last known good values? 
            # If fmid is nan, we can't bisect safely.
            # Try Newton from lo or hi?
            # For now, break and return best available
            break
        
        if abs(fmid) <= 1e-8:
            return mid
        
        # Try Newton step from mid
        h = 1e-8 * max(1.0, abs(mid))
        fmid_h = _f(mid + h)
        newton_taken = False
        if math.isfinite(fmid_h):
            d = (fmid_h - fmid) / h
            if d != 0 and math.isfinite(d):
                x_new = mid - fmid / d
                if math.isfinite(x_new) and lo < x_new < hi:
                    fnew = _f(x_new)
                    if math.isfinite(fnew):
                        if abs(fnew) <= 1e-8:
                            return x_new
                        # Update bracket
                        if flo * fnew < 0:
                            hi = x_new
                            fhi = fnew
                        else:
                            lo = x_new
                            flo = fnew
                        newton_taken = True
        
        if not newton_taken:
            # Bisection step
            if flo * fmid < 0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid
        
        if abs(hi - lo) < 1e-12 * max(1.0, abs(mid)):
            break
    
    # Return best estimate
    candidates = [lo, hi, (lo + hi) / 2.0]
    best_x = lo
    best_f = abs(flo) if math.isfinite(flo) else float('inf')
    for c in candidates:
        if not math.isfinite(c):
            continue
        fc = _f(c)
        if math.isfinite(fc) and abs(fc) < best_f:
            best_f = abs(fc)
            best_x = c
    
    return best_x
# EVOLVE-BLOCK-END
