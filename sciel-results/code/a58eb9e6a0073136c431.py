import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 480
    start_time = time.time()
    DEADLINE = start_time + 140
    
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

    # Initial evaluation
    fx0 = safe_f(x0)
    if abs(fx0) <= TOL:
        return x0
    
    a = x0
    b = None
    fa = fx0
    bracket_found = False
    
    # Phase 1: Quick local scan (8 calls max)
    small_offsets = [0.01, -0.01, 0.05, -0.05, 0.1, -0.1, 0.5, -0.5]
    for d in small_offsets:
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        x_new = x0 + d
        f_new = safe_f(x_new)
        if math.isinf(f_new):
            continue
        if fa * f_new < 0:
            a = x0
            b = x_new
            bracket_found = True
            break
    
    # Phase 2: Spectral-aware period estimation
    if not bracket_found:
        # Try candidate periods corresponding to frequencies k
        # For oscillatory functions, the dominant period is often 2*pi/k
        candidate_periods = []
        for k in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
            period = 2.0 * math.pi / k
            candidate_periods.append(period)
        
        best_period = None
        best_oscillation = 0.0
        
        for period in candidate_periods:
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            x_plus = x0 + period / 4.0
            x_minus = x0 - period / 4.0
            fp = safe_f(x_plus)
            fm = safe_f(x_minus)
            
            if math.isinf(fp) or math.isinf(fm):
                continue
            
            # Measure oscillation: difference from f(x0)
            osc = abs(fp - fa) + abs(fm - fa)
            if osc > best_oscillation:
                best_oscillation = osc
                best_period = period
        
        # Phase 3: Use estimated period to find brackets
        if best_period is not None:
            # Place brackets at multiples of period/2 (half-period = sign change expected)
            half_period = best_period / 2.0
            for n in range(1, 8):
                if calls >= MAX_CALLS or time.time() > DEADLINE:
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
            
            # Also try quarter-period offsets for finer bracketing
            if not bracket_found:
                quarter_period = best_period / 4.0
                for n in range(1, 10):
                    if calls >= MAX_CALLS or time.time() > DEADLINE:
                        break
                    for sign in [1, -1]:
                        x_new = x0 + sign * n * quarter_period
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
        
        # Phase 4: Fallback to geometric expansion if spectral method didn't find bracket
        if not bracket_found:
            for d in [1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0, 50.0, -50.0]:
                if calls >= MAX_CALLS or time.time() > DEADLINE:
                    break
                x_new = x0 + d
                f_new = safe_f(x_new)
                if math.isinf(f_new):
                    continue
                if fa * f_new < 0:
                    a = x0
                    b = x_new
                    bracket_found = True
                    break
    
    if bracket_found:
        try:
            root = brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=30)
            if math.isfinite(root):
                return root
        except Exception:
            pass
        
        # Fallback to bisection if brentq fails
        try:
            lo, hi = min(a, b), max(a, b)
            flo, fhi = safe_f(lo), safe_f(hi)
            if flo * fhi > 0:
                raise ValueError("No sign change")
            for _ in range(40):
                if calls >= MAX_CALLS or time.time() > DEADLINE:
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
    x = best_x
    prev_x = None
    for _ in range(30):
        if calls >= MAX_CALLS or time.time() > DEADLINE:
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
             
        # If we're oscillating, break
        if prev_x is not None and abs(x_new - x) < 1e-10 * max(1.0, abs(x)):
            break
        prev_x = x
        x = x_new
        
    return best_x
