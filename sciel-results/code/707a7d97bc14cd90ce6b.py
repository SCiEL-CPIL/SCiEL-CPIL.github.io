import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x0 = float(x0)
    start_time = time.time()
    deadline = start_time + 115.0
    
    call_count = 0
    max_calls = 500
    best_x = x0
    best_f = float('inf')
    
    def f_counted(x):
        nonlocal call_count, best_x, best_f
        if call_count >= max_calls:
            raise Exception("Max calls exceeded")
        call_count += 1
        val = f(x)
        if abs(val) < abs(best_f):
            best_f = val
            best_x = x
        return val
    
    # Check if x0 is already a root
    fx0 = f_counted(x0)
    if abs(fx0) <= 1e-8:
        return x0
    
    # Strategy 1: Try cubic reconstruction with 4 points
    points = [x0, x0 + 1.0, x0 + 2.0, x0 + 3.0]
    if points[0] == points[1]:
        points = [x0, x0 + 0.1, x0 + 0.2, x0 + 0.3]
    
    try:
        if time.time() > deadline:
            return best_x
        
        vals = []
        # Evaluate and check for root immediately to save calls
        for p in points:
            if time.time() > deadline:
                return best_x
            v = f_counted(p)
            vals.append(v)
            if abs(v) <= 1e-8:
                return p
        
        # Interpolate cubic polynomial
        coeffs = np.polyfit(points, vals, 3)
        roots = np.roots(coeffs)
        
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
        
        if not real_roots:
            raise ValueError("No real roots from cubic")
        
        best_root = min(real_roots, key=lambda r: abs(r - x0))
        
        # Verify the root
        if time.time() > deadline:
            return best_root
        f_best = f_counted(best_root)
        if abs(f_best) <= 1e-8:
            return best_root
            
    except Exception:
        pass
    
    # Strategy 2: Bracketing with Brent's method
    a = x0
    b = x0
    fa = fx0
    fb = fx0
    
    step = 1e-2
    found_bracket = False
    last_a, last_b, last_fa, last_fb = None, None, None, None
    
    for i in range(50):
        if time.time() > deadline or call_count > max_calls - 10:
            break
            
        b = x0 + step
        if time.time() > deadline:
            break
        fb = f_counted(b)
        if abs(fb) <= 1e-8:
            return b
        if fa * fb < 0:
            found_bracket = True
            last_a, last_b, last_fa, last_fb = x0, b, fa, fb
            break
            
        a = x0 - step
        if time.time() > deadline:
            break
        fa = f_counted(a)
        if abs(fa) <= 1e-8:
            return a
        if fa * fx0 < 0:
            found_bracket = True
            last_a, last_b, last_fa, last_fb = a, x0, fa, fx0
            break
            
        step *= 2.0
        
    if found_bracket:
        try:
            root = brentq(f_counted, last_a, last_b, xtol=1e-12, rtol=1e-12, maxiter=100)
            return root
        except Exception:
            pass
    
    # Strategy 3: Secant Method
    x1 = x0 + 1e-4
    if call_count < max_calls - 1:
        fx1 = f_counted(x1)
    else:
        fx1 = fx0
    
    x_prev = x0
    x_curr = x1
    f_prev = fx0
    f_curr = fx1
    
    for _ in range(50):
        if time.time() > deadline or call_count > max_calls - 1:
            break
            
        if abs(f_curr) <= 1e-8:
            return x_curr
        if abs(f_prev) <= 1e-8:
            return x_prev
            
        denom = f_curr - f_prev
        if abs(denom) < 1e-15:
            break
            
        x_new = x_curr - f_curr * (x_curr - x_prev) / denom
        
        if not math.isfinite(x_new):
            break
            
        if abs(x_new - x_curr) > 10 * max(1.0, abs(x_curr)):
             x_new = x_curr + 10 * max(1.0, abs(x_curr)) * (1 if x_new > x_curr else -1)
             
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_new
        
        if call_count < max_calls:
            f_curr = f_counted(x_curr)
        else:
            break
            
        if abs(f_curr) <= 1e-8:
            return x_curr
            
    if abs(f_curr) < abs(f_prev):
        return x_curr
    else:
        return x_prev
