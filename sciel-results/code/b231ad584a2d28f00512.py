import math
import time

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 490
    deadline = time.time() + 140
    
    calls = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def safe_f(x):
        nonlocal calls, best_x, best_abs
        if calls >= MAX_CALLS:
            return float('inf')
        if time.time() > deadline:
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
    
    # Try quick Newton from x0 first (up to 5 iterations, 10 calls)
    x = float(x0)
    for i in range(5):
        if time.time() > deadline or calls >= MAX_CALLS - 5:
            break
        fx = safe_f(x)
        if abs(fx) <= TOL:
            return x
        h = 1e-7 * max(1.0, abs(x))
        fx2 = safe_f(x + h)
        if abs(fx2) <= TOL:
            return x + h
        if fx == float('inf') or fx2 == float('inf'):
            break
        d = (fx2 - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        x = x_new
    
    # Scan for a bracket by expanding outward from x0
    # Try multiple scales to handle oscillatory functions
    step = 0.1
    bracket_found = False
    for scale in [1.0, 0.5, 2.0, 0.25, 4.0, 0.125, 8.0]:
        step = 0.1 * scale
        if time.time() > deadline or calls >= MAX_CALLS - 20:
            break
        
        # Scan in both directions
        for direction in [1, -1]:
            if time.time() > deadline or calls >= MAX_CALLS - 20:
                break
            
            prev_x = float(x0)
            prev_fx = safe_f(prev_x)
            
            if abs(prev_fx) <= TOL:
                return prev_x
            
            for i in range(50):
                if time.time() > deadline or calls >= MAX_CALLS - 10:
                    break
                
                x_try = float(x0) + direction * step * (i + 1)
                fx_try = safe_f(x_try)
                
                if abs(fx_try) <= TOL:
                    return x_try
                
                if prev_fx != float('inf') and fx_try != float('inf'):
                    if prev_fx * fx_try < 0:
                        # Found a bracket! Use Brent's method
                        a, b = min(prev_x, x_try), max(prev_x, x_try)
                        try:
                            from scipy.optimize import brentq
                            # Wrap f to count calls and respect deadline
                            def f_wrap(x):
                                if calls >= MAX_CALLS or time.time() > deadline:
                                    return float('inf')
                                return safe_f(x)
                            
                            root = brentq(f_wrap, a, b, xtol=1e-12, rtol=1e-14, maxiter=50)
                            if math.isfinite(root):
                                return root
                        except Exception:
                            pass
                        bracket_found = True
                        break
                
                prev_x = x_try
                prev_fx = fx_try
    
    # If no bracket found, use Nelder-Mead on |f(x)|
    if not bracket_found and calls < MAX_CALLS - 50:
        try:
            from scipy.optimize import minimize
            def objective(x_vec):
                if calls >= MAX_CALLS or time.time() > deadline:
                    return float('inf')
                x_val = x_vec[0]
                if not math.isfinite(x_val):
                    return float('inf')
                val = safe_f(x_val)
                return abs(val) if math.isfinite(val) else float('inf')
            
            # Initialize simplex around x0
            x0_vec = [float(x0)]
            options = {
                'maxiter': 100,
                'xatol': 1e-12,
                'fatol': 1e-10,
                'adaptive': True
            }
            
            result = minimize(objective, x0_vec, method='Nelder-Mead', options=options)
            if result.success and math.isfinite(result.x[0]):
                x_nm = result.x[0]
                fx_nm = safe_f(x_nm)
                if abs(fx_nm) <= TOL:
                    return x_nm
                if abs(fx_nm) < best_abs:
                    best_x = x_nm
                    best_abs = abs(fx_nm)
        except Exception:
            pass
    
    # If we didn't find a bracket, return best x found
    return best_x
