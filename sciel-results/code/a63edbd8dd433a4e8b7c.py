import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 480
    start_time = time.time()
    DEADLINE = start_time + 145.0

    calls = 0
    best_x = float(x0)
    best_abs = float('inf')

    def safe_f(x):
        nonlocal calls, best_x, best_abs
        if calls >= MAX_CALLS:
            return float('inf')
        if time.time() > DEADLINE:
            return float('inf')
        calls += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            abs_val = abs(val)
            if abs_val < best_abs:
                best_abs = abs_val
                best_x = x
            return val
        except Exception:
            return float('inf')

    def time_ok():
        return time.time() < DEADLINE

    x0 = float(x0)
    fx0 = safe_f(x0)
    if abs(fx0) <= TOL:
        return x0
    if not math.isfinite(fx0):
        # If x0 is bad, try to find a good point quickly
        for dx in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]:
            if calls >= MAX_CALLS or not time_ok():
                break
            x_try = x0 + dx
            fx_try = safe_f(x_try)
            if abs(fx_try) <= TOL:
                return x_try
            if math.isfinite(fx_try):
                break
        if not math.isfinite(fx0):
            return best_x

    a = x0
    b = None
    fa = fx0
    bracket_found = False

    # Phase 1: Multi-scale scanning for sign changes (Inspiration 1 & 3)
    # This is more robust than fixed offsets for finding brackets in oscillatory functions
    scales = [0.1, 0.5, 2.0, 10.0]
    
    for scale in scales:
        if calls >= MAX_CALLS - 20 or not time_ok():
            break
        
        for direction in [1, -1]:
            if calls >= MAX_CALLS - 20 or not time_ok():
                break
            
            # Scan a limited number of steps at this scale
            # We limit steps to save budget for refinement
            max_steps = 50 if scale < 1.0 else 20
            
            prev_x = x0
            prev_f = fa
            
            for i in range(1, max_steps + 1):
                if calls >= MAX_CALLS - 20 or not time_ok():
                    break
                
                x_new = x0 + direction * i * scale
                f_new = safe_f(x_new)
                
                if not math.isfinite(f_new):
                    # If we hit a singularity or bad region, stop this direction/scale
                    # but allow other scales to try
                    break
                
                if abs(f_new) <= TOL:
                    return x_new

                if prev_f * f_new < 0:
                    a = prev_x
                    b = x_new
                    bracket_found = True
                    break
                
                prev_x = x_new
                prev_f = f_new
            
            if bracket_found:
                break
    
    # If no bracket found via simple scanning, try period-based guessing (Current Program Logic)
    if not bracket_found:
        # Try a smaller set of candidate periods
        candidate_periods = []
        for k in [1.0, 2.0, 4.0, 8.0, 16.0]:
            period = 2.0 * math.pi / k
            candidate_periods.append(period)
        
        best_period = None
        best_oscillation = 0.0
        
        for period in candidate_periods:
            if calls >= MAX_CALLS - 20 or not time_ok():
                break
            x_plus = x0 + period / 4.0
            x_minus = x0 - period / 4.0
            fp = safe_f(x_plus)
            fm = safe_f(x_minus)
            
            if math.isinf(fp) or math.isinf(fm):
                continue
            
            osc = abs(fp - fa) + abs(fm - fa)
            if osc > best_oscillation:
                best_oscillation = osc
                best_period = period
        
        if best_period is not None:
            half_period = best_period / 2.0
            for n in range(1, 6):
                if calls >= MAX_CALLS - 20 or not time_ok():
                    break
                for sign in [1, -1]:
                    x_new = x0 + sign * n * half_period
                    f_new = safe_f(x_new)
                    if math.isinf(f_new):
                        continue
                    if fa * f_new < 0:
                        a = x0
                        b = x_new
                        bracket_found = True
                        break
                if bracket_found:
                    break

    if bracket_found:
        # Refinement Phase: Brent's Method (Current Program)
        # Brent is generally more efficient than pure bisection or Newton for bracketed roots
        try:
            root = brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            if math.isfinite(root):
                # Final check
                if abs(safe_f(root)) <= TOL:
                    return root
        except Exception:
            pass
        
        # Fallback to bisection if brentq fails (e.g. due to call limit or non-convergence)
        try:
            lo, hi = min(a, b), max(a, b)
            flo = safe_f(lo)
            fhi = safe_f(hi)
            if flo * fhi > 0:
                raise ValueError("No sign change")
            
            for _ in range(40):
                if calls >= MAX_CALLS - 5 or not time_ok():
                    break
                mid = (lo + hi) / 2
                fmid = safe_f(mid)
                if abs(fmid) <= TOL:
                    return mid
                if flo * fmid < 0:
                    hi = mid
                    fhi = fmid
                else:
                    lo = mid
                    flo = fmid
            return (lo + hi) / 2
        except Exception:
            pass

    # If no bracket found, try Newton's method from the best point found
    # This handles cases where the root is not bracketed by our scan (e.g. tangential)
    # or if the function is very steep.
    x = best_x
    prev_x = None
    for _ in range(30):
        if calls >= MAX_CALLS - 5 or not time_ok():
            break
        fx = safe_f(x)
        if abs(fx) <= TOL:
            return x
        
        h = 1e-7 * max(1.0, abs(x))
        f_minus = safe_f(x - h)
        f_plus = safe_f(x + h)
        if math.isinf(f_minus) or math.isinf(f_plus):
            break
        d = (f_plus - f_minus) / (2 * h)
        if abs(d) < 1e-15 or not math.isfinite(d):
            break
        
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
            
        # Damping if step is too large
        max_step = 5.0 * max(1.0, abs(x))
        if abs(x_new - x) > max_step:
             direction = 1 if x_new > x else -1
             x_new = x + direction * max_step
             
        if prev_x is not None and abs(x_new - x) < 1e-10 * max(1.0, abs(x)):
            break
        prev_x = x
        x = x_new
        
    return best_x
