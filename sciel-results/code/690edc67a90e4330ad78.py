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
    
    # Strategy 1: Try to bracket the root by scanning outward with increasing steps
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
    
    # Strategy 2: Newton's method with damped steps (from current program)
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
        step_newton = fx / d
        # Damp the step to avoid divergence
        if abs(step_newton) > 10 * max(1.0, abs(x)):
            step_newton = 10 * max(1.0, abs(x)) * math.copysign(1, step_newton)
        x_new = x - step_newton
        if not math.isfinite(x_new):
            break
        x = x_new
    
    if best_abs <= TOL:
        return best_x
    
    # Strategy 3: Secant method from multiple starting points (inspired by inspiration 1)
    for x1_init in [x0, x0 + 1e-4, x0 - 1e-4, x0 + 1e-2, x0 - 1e-2, x0 + 0.1, x0 - 0.1]:
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        x1 = float(x1_init)
        try:
            f1 = safe_f(x1)
            if not math.isfinite(f1):
                continue
        except Exception:
            continue
        
        x0_sec = float(x0)
        f0 = safe_f(x0_sec)
        if not math.isfinite(f0) or not math.isfinite(f1) or f0 == f1:
            continue
        
        for i in range(30):
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            try:
                denom = f1 - f0
                if denom == 0:
                    break
                x_new = x1 - f1 * (x1 - x0_sec) / denom
                if not math.isfinite(x_new):
                    break
                f_new = safe_f(x_new)
                if not math.isfinite(f_new):
                    break
                if abs(f_new) <= TOL:
                    return x_new
                x0_sec = x1
                f0 = f1
                x1 = x_new
                f1 = f_new
            except Exception:
                break
    
    if best_abs <= TOL:
        return best_x
    
    # Strategy 4: Halley's method with finite difference (inspired by inspiration 1)
    x = float(x0)
    for i in range(20):
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        try:
            fx = safe_f(x)
            if not math.isfinite(fx):
                break
            if abs(fx) <= TOL:
                return x
            
            h = 1e-5 * max(1.0, abs(x))
            fx_plus = safe_f(x + h)
            fx_minus = safe_f(x - h)
            if not math.isfinite(fx_plus) or not math.isfinite(fx_minus):
                break
            
            d1 = (fx_plus - fx_minus) / (2 * h)
            d2 = (fx_plus - 2 * fx + fx_minus) / (h * h)
            
            if d1 == 0 or not math.isfinite(d1):
                break
            
            if d2 != 0:
                denom = 1.0 - 0.5 * fx * d2 / d1
                if abs(denom) < 1e-12:
                    break
                dx = fx / (d1 * denom)
            else:
                dx = fx / d1
            
            max_step = max(1.0, abs(x)) * 0.3
            if abs(dx) > max_step:
                dx = max_step * math.copysign(1.0, dx)
            x_new = x + dx
            if not math.isfinite(x_new):
                break
            x = x_new
        except Exception:
            break
    
    if best_abs <= TOL:
        return best_x
    
    # Final check
    if calls < MAX_CALLS and time.time() < DEADLINE:
        fx_best = safe_f(best_x)
        if abs(fx_best) <= TOL:
            return best_x
    
    return best_x
