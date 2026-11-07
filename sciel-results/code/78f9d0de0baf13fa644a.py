import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 120.0  # 120s safety margin
    calls = 0
    max_calls = 480
    best_x = float('nan')
    best_val = float('inf')
    
    def _f(x):
        nonlocal calls, best_x, best_val
        if calls >= max_calls:
            return float('nan')
        if time.time() > deadline:
            return float('nan')
        calls += 1
        try:
            val = f(x)
            if not math.isnan(val) and not math.isinf(val):
                if abs(val) < best_val:
                    best_val = abs(val)
                    best_x = x
            return val
        except:
            return float('nan')

    def time_ok():
        return time.time() < deadline

    x = float(x0)
    fx = _f(x)
    if math.isnan(fx):
        return best_x if not math.isnan(best_x) else x
    if abs(fx) <= 1e-8: return x

    # 1. Find Bracket
    a, b, fa, fb = None, None, None, None
    base_step = 1.0
    if abs(x) > 1e6: base_step = 1e6
    elif abs(x) > 1e3: base_step = 1e3
    elif abs(x) > 1: base_step = 1.0
    else: base_step = 0.1
    
    found_bracket = False
    # Try positive
    step = base_step * 0.1
    for i in range(1, 20):
        if not time_ok() or calls >= max_calls: break
        xp = x + step
        fp = _f(xp)
        if math.isnan(fp): break
        if abs(fp) <= 1e-8: return xp
        if fx * fp < 0:
            a, b, fa, fb = x, xp, fx, fp
            found_bracket = True
            break
        step *= 1.5
        
    if not found_bracket:
        # Try negative
        step = base_step * 0.1
        for i in range(1, 20):
            if not time_ok() or calls >= max_calls: break
            xn = x - step
            fn = _f(xn)
            if math.isnan(fn): break
            if abs(fn) <= 1e-8: return xn
            if fx * fn < 0:
                a, b, fa, fb = xn, x, fn, fx
                found_bracket = True
                break
            step *= 1.5

    if found_bracket:
        # 2. Refine using Guarded Secant / Illinois
        # a < b, fa * fb < 0
        for i in range(100):
            if calls >= max_calls or not time_ok(): break
            if abs(fa) <= 1e-8: return a
            if abs(fb) <= 1e-8: return b
            
            # Try Secant
            if fa == fb:
                m = (a+b)/2
            else:
                m = (a*fb - b*fa) / (fb - fa)
                
            if a < m < b:
                fm = _f(m)
                if math.isnan(fm): break
                if abs(fm) <= 1e-8: return m
                if fa * fm < 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            else:
                # Bisection
                m = (a+b)/2
                fm = _f(m)
                if math.isnan(fm): break
                if abs(fm) <= 1e-8: return m
                if fa * fm < 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            
        # After refinement, check endpoints
        if abs(fa) <= 1e-8: return a
        if abs(fb) <= 1e-8: return b
        return a if abs(fa) < abs(fb) else b

    # No bracket found - try local minimization of (f(x))^2
    if not time_ok() or calls >= max_calls:
        return best_x if not math.isnan(best_x) else x

    # Define objective: minimize (f(x))^2
    def objective(x_val):
        if calls >= max_calls or not time_ok():
            return float('inf')
        try:
            val = f(x_val)
            if math.isnan(val) or math.isinf(val):
                return float('inf')
            return val * val
        except:
            return float('inf')
    
    # Determine search bounds for minimize_scalar
    search_radius = base_step * 10
    if abs(x) > 1e6:
        search_radius = 1e6
    elif abs(x) > 1e3:
        search_radius = 1e3
    
    lower = x - search_radius
    upper = x + search_radius
    
    try:
        result = minimize_scalar(objective, bounds=(lower, upper), method='bounded',
                                 options={'maxiter': 50, 'xatol': 1e-10})
        if result.success and result.fun <= 1e-16:  # (1e-8)^2 = 1e-16
            return float(result.x)
        elif result.x is not None and not math.isnan(result.x):
            # Check if this is better than best_x
            try:
                fx_result = f(result.x)
                if not math.isnan(fx_result) and not math.isinf(fx_result):
                    if abs(fx_result) < best_val:
                        best_val = abs(fx_result)
                        best_x = result.x
                    if abs(fx_result) <= 1e-8:
                        return float(result.x)
            except:
                pass
    except:
        pass

    # Final fallback: return best_x found
    if not math.isnan(best_x):
        return best_x
    return x
