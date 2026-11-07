import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 490  # Leave margin for safety
    DEADLINE = time.time() + 140  # 140s out of 150s limit
    
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
    
    x = float(x0)
    fx = safe_f(x)
    if abs(fx) <= TOL:
        return x
    
    # Try to bracket the root by scanning outward
    # Start with small steps, then increase
    step = 0.1 * max(1.0, abs(x))
    found_bracket = False
    bracket_a = x
    bracket_b = None
    
    # Scan to the right
    for i in range(50):
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        x_try = x + step
        f_try = safe_f(x_try)
        if math.isfinite(f_try):
            if fx * f_try < 0:
                bracket_a = x
                bracket_b = x_try
                found_bracket = True
                break
            fx = f_try
            x = x_try
        else:
            break
        step *= 1.5
    
    if not found_bracket:
        # Scan to the left from original x0
        x = float(x0)
        fx = safe_f(x)
        step = 0.1 * max(1.0, abs(x))
        for i in range(50):
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            x_try = x - step
            f_try = safe_f(x_try)
            if math.isfinite(f_try):
                if fx * f_try < 0:
                    bracket_a = x_try
                    bracket_b = x
                    found_bracket = True
                    break
                fx = f_try
                x = x_try
            else:
                break
            step *= 1.5
    
    if found_bracket:
        try:
            result = brentq(f, bracket_a, bracket_b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if math.isfinite(result):
                return result
        except Exception:
            pass
    
    # Fallback: try Newton's method with damped steps
    x = float(x0)
    for _ in range(30):
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        fx = safe_f(x)
        if abs(fx) <= TOL:
            return x
        h = 1e-6 * max(1.0, abs(x))
        if calls >= MAX_CALLS:
            break
        fx_plus = safe_f(x + h)
        if not math.isfinite(fx_plus):
            break
        d = (fx_plus - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        step = fx / d
        # Damp the step to avoid divergence
        if abs(step) > 10 * max(1.0, abs(x)):
            step = 10 * max(1.0, abs(x)) * math.copysign(1, step)
        x_new = x - step
        if not math.isfinite(x_new):
            break
        x = x_new
    
    if calls >= MAX_CALLS or time.time() > DEADLINE:
        return best_x
    
    # Final check
    fx_best = safe_f(best_x)
    if abs(fx_best) <= TOL:
        return best_x
    
    return best_x
