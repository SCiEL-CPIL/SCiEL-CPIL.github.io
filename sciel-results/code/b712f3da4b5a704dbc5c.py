"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Find a root of f near x0 using a combination of:
    1. Quick local check
    2. Adaptive bracketing with geometric expansion
    3. Bisection + Newton hybrid for convergence
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
    
    # Step 1: Check if x0 is already a root
    fx0 = _f(x0)
    if math.isfinite(fx0) and abs(fx0) <= 1e-8:
        return x0
    
    # Step 2: Try a few local steps with Newton's method (finite difference)
    x = x0
    fx = fx0
    for _ in range(10):
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
    
    # Step 3: Adaptive bracketing - expand geometrically in both directions
    # to find a sign change bracket [a, b]
    a = x0
    b = x0
    fa = fx0
    fb = fx0
    
    # Try expanding with multiple step sizes to catch oscillatory functions
    step = max(1e-3, abs(x0) * 1e-3)
    found_bracket = False
    
    # First, try small steps in both directions
    for direction in [1, -1]:
        for i in range(20):
            xi = x0 + direction * step * (2 ** i)
            fi = _f(xi)
            if not math.isfinite(fi):
                continue
            if abs(fi) <= 1e-8:
                return xi
            # Check for sign change with x0
            if math.isfinite(fx0) and fx0 * fi < 0:
                if direction == 1:
                    a, b = x0, xi
                    fa, fb = fx0, fi
                else:
                    a, b = xi, x0
                    fa, fb = fi, fx0
                found_bracket = True
                break
            # Also check sign change between consecutive points
            if i > 0:
                x_prev = x0 + direction * step * (2 ** (i - 1))
                # We need f(x_prev), but we didn't store it. Let's restructure.
                pass
        if found_bracket:
            break
    
    # If no bracket found with geometric expansion from x0, try a denser sweep
    if not found_bracket:
        # Sweep around x0 with finer granularity
        base_step = max(1e-4, abs(x0) * 1e-4)
        prev_x = None
        prev_f = None
        for i in range(-50, 51):
            xi = x0 + i * base_step
            fi = _f(xi)
            if not math.isfinite(fi):
                prev_x = None
                prev_f = None
                continue
            if abs(fi) <= 1e-8:
                return xi
            if prev_f is not None and math.isfinite(prev_f) and prev_f * fi < 0:
                a, b = prev_x, xi
                fa, fb = prev_f, fi
                found_bracket = True
                break
            prev_x = xi
            prev_f = fi
    
    if not found_bracket:
        # Last resort: try larger geometric expansion
        step = max(1.0, abs(x0))
        for direction in [1, -1]:
            for i in range(15):
                xi = x0 + direction * step * (2 ** i)
                fi = _f(xi)
                if not math.isfinite(fi):
                    continue
                if abs(fi) <= 1e-8:
                    return xi
                if math.isfinite(fx0) and fx0 * fi < 0:
                    if direction == 1:
                        a, b = x0, xi
                        fa, fb = fx0, fi
                    else:
                        a, b = xi, x0
                        fa, fb = fi, fx0
                    found_bracket = True
                    break
            if found_bracket:
                break
    
    if not found_bracket:
        # If we still can't find a bracket, return the best we have
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
    
    # Step 4: Bisection + Newton hybrid to converge within the bracket
    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
    
    # Bisection with Newton acceleration
    lo, hi = a, b
    flo, fhi = fa, fb
    
    for _ in range(200):
        if not math.isfinite(flo) or not math.isfinite(fhi):
            break
        if abs(flo) <= 1e-8:
            return lo
        if abs(fhi) <= 1e-8:
            return hi
        
        # Try Newton step from midpoint or current best
        mid = (lo + hi) / 2.0
        fmid = _f(mid)
        if not math.isfinite(fmid):
            # Fall back to pure bisection
            if flo * fmid < 0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid
            continue
        
        if abs(fmid) <= 1e-8:
            return mid
        
        # Try Newton step
        h = 1e-8 * max(1.0, abs(mid))
        fmid_h = _f(mid + h)
        if math.isfinite(fmid_h):
            d = (fmid_h - fmid) / h
            if d != 0 and math.isfinite(d):
                x_new = mid - fmid / d
                if math.isfinite(x_new) and lo < x_new < hi:
                    fnew = _f(x_new)
                    if math.isfinite(fnew):
                        if abs(fnew) <= 1e-8:
                            return x_new
                        # Update bracket based on sign
                        if flo * fnew < 0:
                            hi = x_new
                            fhi = fnew
                        else:
                            lo = x_new
                            flo = fnew
                        continue
        
        # Bisection step
        if flo * fmid < 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
        
        # Check convergence
        if abs(hi - lo) < 1e-12 * max(1.0, abs(mid)):
            break
    
    # Return the best estimate
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
