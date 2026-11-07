import math
import time
from scipy.optimize import brentq, minimize

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 490
    DEADLINE = time.time() + 140
    
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
    
    # Strategy 1: Multi-scale bracketing scan
    found_bracket = False
    bracket_a = None
    bracket_b = None
    
    # Try multiple scales for initial step to handle oscillatory functions
    scales = [0.1, 0.5, 2.0, 5.0, 10.0]
    
    for scale in scales:
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        
        step = scale * max(1.0, abs(x0))
        
        # Scan right
        prev_x = x
        prev_fx = fx
        
        for i in range(25):
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            x_try = x0 + step
            f_try = safe_f(x_try)
            if not math.isfinite(f_try):
                break
            if prev_fx * f_try < 0:
                bracket_a = prev_x
                bracket_b = x_try
                found_bracket = True
                break
            prev_x = x_try
            prev_fx = f_try
            step *= 1.8
        
        if found_bracket:
            break
        
        # Scan left
        step = scale * max(1.0, abs(x0))
        prev_x = x
        prev_fx = fx
        
        for i in range(25):
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            x_try = x0 - step
            f_try = safe_f(x_try)
            if not math.isfinite(f_try):
                break
            if prev_fx * f_try < 0:
                bracket_a = x_try
                bracket_b = prev_x
                found_bracket = True
                break
            prev_x = x_try
            prev_fx = f_try
            step *= 1.8
        
        if found_bracket:
            break
    
    if found_bracket:
        try:
            result = brentq(safe_f, bracket_a, bracket_b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if math.isfinite(result):
                if calls < MAX_CALLS:
                    f_result = safe_f(result)
                    if abs(f_result) <= TOL:
                        return result
                    if math.isfinite(f_result):
                        return result
                return result
        except Exception:
            pass
    
    # Strategy 2: Nelder-Mead on |f(x)| - handles oscillatory landscapes without brackets
    remaining_calls = MAX_CALLS - calls
    if remaining_calls > 30 and time.time() < DEADLINE - 10:
        def objective(x_vec):
            return safe_f(x_vec[0])
        
        try:
            initial_simplex = [
                [x0],
                [x0 + 0.1 * max(1.0, abs(x0))],
                [x0 - 0.1 * max(1.0, abs(x0))]
            ]
            result = minimize(objective, [x0], method='Nelder-Mead',
                             options={'maxiter': 150, 'maxfev': min(remaining_calls - 5, 200),
                                      'xatol': 1e-12, 'fatol': 1e-12,
                                      'initial_simplex': initial_simplex})
            if result.x is not None and len(result.x) > 0 and math.isfinite(result.x[0]):
                f_val = safe_f(result.x[0])
                if abs(f_val) <= TOL:
                    return result.x[0]
                if math.isfinite(f_val) and abs(f_val) < best_abs:
                    best_x = result.x[0]
                    best_abs = abs(f_val)
        except Exception:
            pass
    
    # Strategy 3: Newton's method with damped steps
    x = float(x0)
    for _ in range(20):
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        fx = safe_f(x)
        if abs(fx) <= TOL:
            return x
        if not math.isfinite(fx):
            break
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
        max_step = 5.0 * max(1.0, abs(x))
        if abs(step_newton) > max_step:
            step_newton = max_step * math.copysign(1, step_newton)
        x_new = x - step_newton
        if not math.isfinite(x_new):
            break
        x = x_new
    
    if best_abs <= TOL:
        return best_x
    
    # Strategy 4: Secant method
    if calls < MAX_CALLS:
        x1 = float(x0)
        f1 = safe_f(x1)
        if math.isfinite(f1):
            x2 = x1 + 1e-4 * max(1.0, abs(x1))
            f2 = safe_f(x2)
            if math.isfinite(f2) and f1 != f2:
                for i in range(20):
                    if calls >= MAX_CALLS or time.time() > DEADLINE:
                        break
                    denom = f2 - f1
                    if denom == 0:
                        break
                    x_new = x2 - f2 * (x2 - x1) / denom
                    if not math.isfinite(x_new):
                        break
                    f_new = safe_f(x_new)
                    if not math.isfinite(f_new):
                        break
                    if abs(f_new) <= TOL:
                        return x_new
                    x1 = x2
                    f1 = f2
                    x2 = x_new
                    f2 = f_new
    
    if best_abs <= TOL:
        return best_x
    
    # Final polish pass
    if best_abs < 1e-6 and calls < MAX_CALLS - 5:
        x = best_x
        fx = safe_f(x)
        if math.isfinite(fx) and abs(fx) > TOL:
            for _ in range(5):
                if calls >= MAX_CALLS or time.time() > DEADLINE:
                    break
                h = 1e-7 * max(1.0, abs(x))
                fx_plus = safe_f(x + h)
                if not math.isfinite(fx_plus):
                    break
                d = (fx_plus - fx) / h
                if d == 0 or not math.isfinite(d):
                    break
                step_n = fx / d
                if abs(step_n) > 1e-3 * max(1.0, abs(x)):
                    step_n = 1e-3 * max(1.0, abs(x)) * math.copysign(1, step_n)
                x_new = x - step_n
                if not math.isfinite(x_new):
                    break
                fx_new = safe_f(x_new)
                if not math.isfinite(fx_new):
                    break
                if abs(fx_new) <= TOL:
                    return x_new
                x = x_new
                fx = fx_new
    
    if calls < MAX_CALLS and time.time() < DEADLINE:
        fx_best = safe_f(best_x)
        if abs(fx_best) <= TOL:
            return best_x
    
    return best_x
