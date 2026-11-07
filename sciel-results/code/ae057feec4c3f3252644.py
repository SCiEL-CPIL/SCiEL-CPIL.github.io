"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Find a root of f near x0.
    """
    start_time = time.time()
    DEADLINE = 110.0
    
    def time_ok():
        return (time.time() - start_time) < DEADLINE

    x = float(x0)
    try:
        fx = f(x)
    except Exception:
        return x
        
    if not math.isfinite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x

    # 1. Find a bracket [a, b] such that f(a)*f(b) <= 0
    a = x
    fa = fx
    step = 1.0
    b = None
    fb = None
    
    # Try expanding in both directions
    for i in range(50):
        if not time_ok():
            break
            
        # Try positive direction
        x_plus = a + step
        try:
            f_plus = f(x_plus)
        except Exception:
            f_plus = float('nan')
            
        if math.isfinite(f_plus):
            if fa * f_plus <= 0:
                b = x_plus
                fb = f_plus
                break
        
        # Try negative direction
        x_minus = a - step
        try:
            f_minus = f(x_minus)
        except Exception:
            f_minus = float('nan')
            
        if math.isfinite(f_minus):
            if fa * f_minus <= 0:
                b = x_minus
                fb = f_minus
                break
        
        step *= 2.0
        if step > 1e10:
            break

    if b is not None:
        # Use Brent's method
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if math.isfinite(root):
                return root
        except Exception:
            pass

    # Fallback: Newton's method
    x = float(x0)
    fx = f(x)
    for i in range(100):
        if not time_ok():
            break
        if abs(fx) <= 1e-8:
            return x
            
        h = 1e-6 * max(1.0, abs(x))
        try:
            fx_h = f(x + h)
        except Exception:
            break
            
        if not math.isfinite(fx_h):
            break
            
        d = (fx_h - fx) / h
        if not math.isfinite(d) or d == 0:
            break
            
        x_new = x - fx / d
        if abs(x_new - x) > 1.0:
             x_new = x + 1.0 if x_new > x else x - 1.0
             
        try:
            fx_new = f(x_new)
        except Exception:
            break
            
        if not math.isfinite(fx_new):
            break
            
        if abs(fx_new) < abs(fx):
            x = x_new
            fx = fx_new
        else:
            x_new = x - 0.5 * (fx / d)
            try:
                fx_new = f(x_new)
            except Exception:
                break
            if math.isfinite(fx_new) and abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
            else:
                break

    return x
# EVOLVE-BLOCK-END
