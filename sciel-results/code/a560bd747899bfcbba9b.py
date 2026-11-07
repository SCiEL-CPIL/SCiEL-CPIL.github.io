"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Find a root of f near x0 using spectral-aware bracketing.
    1. Estimate local period via high-resolution scan.
    2. Use period-based step size to find a bracket.
    3. Converge using Bisection + Newton hybrid.
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

    # Step 2: Estimate local period
    # Scan a small window around x0 to find the smallest distance between sign changes
    # This helps adapt to high-frequency oscillations
    scan_step = 1e-4
    max_scan_points = 200
    sign_changes = []
    prev_x = x0
    prev_f = fx0
    
    for i in range(1, max_scan_points + 1):
        xi = x0 + i * scan_step
        fi = _f(xi)
        if not math.isfinite(fi):
            break
        if prev_f * fi < 0:
            sign_changes.append(xi - prev_x)
        prev_x = xi
        prev_f = fi
        
    # Also scan backwards
    prev_x = x0
    prev_f = fx0
    for i in range(1, max_scan_points + 1):
        xi = x0 - i * scan_step
        fi = _f(xi)
        if not math.isfinite(fi):
            break
        if prev_f * fi < 0:
            sign_changes.append(abs(xi - prev_x))
        prev_x = xi
        prev_f = fi

    if sign_changes:
        min_dist = min(sign_changes)
        # The distance between sign changes is roughly half the period
        # Use a step size smaller than this to ensure we catch the bracket
        bracket_step = max(1e-6, min_dist / 4.0)
    else:
        # No sign changes found in local scan, assume a default scale
        # Scale based on x0 magnitude
        bracket_step = max(1e-3, abs(x0) * 1e-3)

    # Step 3: Find a bracket [a, b] with f(a)*f(b) < 0
    # Search in both directions from x0 using the estimated step size
    a = x0
    b = x0
    fa = fx0
    fb = fx0
    found_bracket = False
    
    # We will search up to a certain number of steps in each direction
    max_steps = 1000
    
    # Try forward
    for i in range(1, max_steps + 1):
        xi = x0 + i * bracket_step
        fi = _f(xi)
        if not math.isfinite(fi):
            continue
        if abs(fi) <= 1e-8:
            return xi
        if math.isfinite(fa) and fa * fi < 0:
            a, b = x0, xi
            fa, fb = fx0, fi
            found_bracket = True
            break
    
    if not found_bracket:
        # Try backward
        for i in range(1, max_steps + 1):
            xi = x0 - i * bracket_step
            fi = _f(xi)
            if not math.isfinite(fi):
                continue
            if abs(fi) <= 1e-8:
                return xi
            if math.isfinite(fa) and fa * fi < 0:
                a, b = xi, x0
                fa, fb = fi, fx0
                found_bracket = True
                break

    # If no bracket found with fine step, try geometric expansion as fallback
    if not found_bracket:
        step = max(1e-3, abs(x0) * 1e-3)
        for direction in [1, -1]:
            for i in range(30):
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
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
    
    lo, hi = a, b
    flo, fhi = fa, fb

    for _ in range(200):
        if not math.isfinite(flo) or not math.isfinite(fhi):
            break
        if abs(flo) <= 1e-8:
            return lo
        if abs(fhi) <= 1e-8:
            return hi
        
        mid = (lo + hi) / 2.0
        fmid = _f(mid)
        if not math.isfinite(fmid):
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
                        if flo * fnew < 0:
                            hi = x_new
                            fhi = fnew
                        else:
                            lo = x_new
                            flo = fnew
                        continue
        
        if flo * fmid < 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
        
        if abs(hi - lo) < 1e-12 * max(1.0, abs(mid)):
            break
    
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
