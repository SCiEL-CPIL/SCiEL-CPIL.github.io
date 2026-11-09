import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def time_ok():
        return (time.time() - start_time) < deadline
    
    x = float(x0)
    if not math.isfinite(x):
        x = 0.0
    
    # Try to get a good initial point
    best_x = x
    best_fx = f(x)
    if not math.isfinite(best_fx):
        for dx in [1.0, -1.0, 0.1, -0.1, 0.01, -0.01, 10.0, -10.0, 0.5, -0.5]:
            x_try = x + dx
            try:
                fx_try = f(x_try)
                if math.isfinite(fx_try):
                    best_x = x_try
                    best_fx = fx_try
                    break
            except:
                pass
    
    x = best_x
    fx = best_fx
    
    if abs(fx) <= 1e-8:
        return x
    
    # Try to find a bracket [a, b] where f(a) and f(b) have opposite signs
    a = None
    b = None
    
    # First check if x0 itself is in a bracket by expanding outward
    step = 1.0
    for _ in range(20):
        if not time_ok():
            break
        for sign in [1, -1]:
            x_try = x + sign * step
            try:
                fx_try = f(x_try)
                if math.isfinite(fx_try):
                    if fx_try == 0:
                        return x_try
                    if fx * fx_try < 0:
                        if sign == 1:
                            a, b = x, x_try
                        else:
                            a, b = x_try, x
                        break
            except:
                pass
        if a is not None:
            break
        step *= 2.0
    
    if a is None:
        # Try some fixed offsets
        for dx in [0.5, -0.5, 2.0, -2.0, 5.0, -5.0, 1.5, -1.5, 3.0, -3.0]:
            if not time_ok():
                break
            x_try = x + dx
            try:
                fx_try = f(x_try)
                if math.isfinite(fx_try) and fx * fx_try < 0:
                    if fx_try > 0 and fx < 0:
                        a, b = x_try, x
                    else:
                        a, b = x, x_try
                    break
            except:
                pass
    
    # If we have a bracket, use Brent's method
    if a is not None and b is not None:
        fa = fx if a == x else f(a)
        fb = f(b) if b != x else fx
        
        # Ensure fa and fb have opposite signs
        if fa * fb > 0:
            # Try to fix the bracket
            mid = (a + b) / 2.0
            fmid = f(mid)
            if math.isfinite(fmid):
                if fa * fmid < 0:
                    b = mid
                    fb = fmid
                elif fmid * fb < 0:
                    a = mid
                    fa = fmid
        
        if fa * fb <= 0:
            try:
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                if math.isfinite(root) and abs(f(root)) <= 1e-8:
                    return root
            except:
                pass
        
        # Fallback to bisection if brentq failed
        for _ in range(100):
            if not time_ok():
                break
            
            mid = (a + b) / 2.0
            fmid = f(mid)
            
            if abs(fmid) <= 1e-8:
                return mid
            if fmid == 0:
                return mid
            
            if fa * fmid < 0:
                b = mid
                fb = fmid
            else:
                a = mid
                fa = fmid
            
            if abs(b - a) < 1e-12:
                break
        
        return (a + b) / 2.0
    
    # No bracket found - use safeguarded Newton/secant method
    best_x = x
    best_fx = fx
    
    x_prev = None
    fx_prev = None
    
    for _ in range(80):
        if not time_ok():
            break
        
        # Compute derivative via finite difference
        h = 1e-7 * max(1.0, abs(x))
        try:
            fx_h = f(x + h)
        except:
            break
        
        if not math.isfinite(fx_h):
            break
        
        d = (fx_h - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        
        # Newton step
        x_new = x - fx / d
        
        # Safeguard: limit step size
        max_step = 10.0 * max(1.0, abs(x))
        if abs(x_new - x) > max_step:
            x_new = x + math.copysign(max_step, x_new - x)
        
        if not math.isfinite(x_new):
            break
        
        try:
            fx_new = f(x_new)
        except:
            break
        
        if not math.isfinite(fx_new):
            break
        
        if abs(fx_new) < abs(fx):
            x, fx = x_new, fx_new
            if abs(fx) <= 1e-8:
                return x
            x_prev, fx_prev = x, fx
        else:
            # Step was bad, try bisection-like approach
            if x_prev is not None and fx_prev is not None:
                # Use secant between x and x_prev
                if x != x_prev and fx != fx_prev:
                    x_sec = x - fx * (x - x_prev) / (fx - fx_prev)
                    if math.isfinite(x_sec):
                        try:
                            fx_sec = f(x_sec)
                            if math.isfinite(fx_sec) and abs(fx_sec) < abs(fx):
                                x, fx = x_sec, fx_sec
                                if abs(fx) <= 1e-8:
                                    return x
                                x_prev, fx_prev = x, fx
                                continue
                        except:
                            pass
            # Shrink step
            h_shrink = 0.5 * max(1.0, abs(x))
            for shrink in [0.5, 0.25, 0.1]:
                x_try = x + shrink * h_shrink * (1 if fx > 0 else -1)
                try:
                    fx_try = f(x_try)
                    if math.isfinite(fx_try) and abs(fx_try) < abs(fx):
                        x, fx = x_try, fx_try
                        x_prev, fx_prev = x, fx
                        break
                except:
                    pass
            else:
                # Stuck, break
                break
        
        if abs(x - x_prev) < 1e-12 if x_prev is not None else False:
            break
    
    return x
