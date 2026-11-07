import math
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Finds a root of f starting from x0.
    Returns x such that |f(x)| <= 1e-8.
    """
    x = float(x0)
    
    # Initial evaluation
    try:
        fx = f(x)
    except Exception:
        return x # Return x0 if f fails immediately, though this will likely fail the check
        
    if not math.isfinite(fx):
        return x
        
    if abs(fx) <= 1e-8:
        return x

    # Strategy 0: Newton-Raphson with numerical derivative
    # Since the functions are smooth polynomials, Newton's method is very fast if x0 is close.
    # We use central difference for derivative estimation.
    h = 1e-5 * max(1.0, abs(x))
    current_x = x
    current_fx = fx
    
    for _ in range(5): # Max 5 Newton steps
        # Estimate derivative
        try:
            fx_plus = f(current_x + h)
            fx_minus = f(current_x - h)
        except Exception:
            break
            
        if not (math.isfinite(fx_plus) and math.isfinite(fx_minus)):
            break
            
        df = (fx_plus - fx_minus) / (2 * h)
        
        if abs(df) < 1e-12:
            break # Derivative too small, Newton will fail
            
        # Newton step
        x_new = current_x - current_fx / df
        
        if not math.isfinite(x_new):
            break
            
        try:
            fx_new = f(x_new)
        except Exception:
            break
            
        if not math.isfinite(fx_new):
            break
            
        # Check for convergence
        if abs(fx_new) <= 1e-8:
            return x_new
            
        # Check for progress: if we didn't improve significantly, abort Newton
        if abs(fx_new) > abs(current_fx) * 0.9:
            break

        current_x = x_new
        current_fx = fx_new
        
    if abs(current_fx) <= 1e-8:
        return current_x

    # If Newton didn't solve it, try to bracket using the best point found or original x0
    # Use the best point from Newton as the center for expansion if it's better
    center_x = current_x if abs(current_fx) < abs(fx) else x
    center_fx = current_fx if abs(current_fx) < abs(fx) else fx

    # Strategy 1: Find a bracket and use Brent's method
    # Try expanding around center_x
    a, b = None, None
    fa = center_fx
    step = 1.0
    max_expansions = 20
    
    for _ in range(max_expansions):
        # Try center_x + step
        x_plus = center_x + step
        try:
            fx_plus = f(x_plus)
            if math.isfinite(fx_plus) and fa * fx_plus < 0:
                a, b = center_x, x_plus
                break
        except Exception:
            pass
            
        # Try center_x - step
        x_minus = center_x - step
        try:
            fx_minus = f(x_minus)
            if math.isfinite(fx_minus) and fa * fx_minus < 0:
                a, b = x_minus, center_x
                break
        except Exception:
            pass
            
        step *= 2.0

    if a is not None and b is not None:
        try:
            # Use looser tolerance to save calls
            root = brentq(f, a, b, xtol=1e-10, rtol=1e-10, maxiter=50)
            if math.isfinite(root):
                return root
        except Exception:
            pass # Fall through to fallback

    # Strategy 2: Fallback to Robust Secant (Illinois Algorithm)
    # Use the best point from Newton as the starting point for Secant
    x1 = center_x
    f1 = center_fx
    x2 = x1 + 1e-6 * max(1.0, abs(x1))
    
    try:
        f2 = f(x2)
    except Exception:
        return x1
        
    if not math.isfinite(f2):
        return x1

    # Illinois Algorithm
    for _ in range(100):
        if abs(f1) <= 1e-8:
            return x1
        if abs(f2) <= 1e-8:
            return x2
            
        if f2 == f1:
            # Degenerate case, perturb
            x2 = x1 + 1e-6
            try:
                f2 = f(x2)
            except Exception:
                return x1
            if not math.isfinite(f2):
                return x1
            continue
            
        x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
        
        # Check for non-finite
        if not math.isfinite(x_new):
            x_new = (x1 + x2) / 2.0
            
        try:
            f_new = f(x_new)
        except Exception:
            return x1
            
        if not math.isfinite(f_new):
            x_new = (x1 + x2) / 2.0
            try:
                f_new = f(x_new)
            except Exception:
                return x1
            if not math.isfinite(f_new):
                return x1

        # Illinois damping: if the new point is on the same side of x1 as x2,
        # and f_new has the same sign as f2 (relative to f1), halve the step
        # Standard Illinois logic:
        # If (x_new - x1) * (x2 - x1) > 0 and (f_new - f1) * (f2 - f1) > 0:
        #    x2 = (x1 + x2) / 2
        # Else:
        #    x2 = x1, f2 = f1
        #    x1 = x_new, f1 = f_new
        
        if (x_new - x1) * (x2 - x1) > 0 and (f_new - f1) * (f2 - f1) > 0:
            x2 = (x1 + x2) / 2.0
            try:
                f2 = f(x2)
            except Exception:
                return x1
            if not math.isfinite(f2):
                return x1
        else:
            x2 = x1
            f2 = f1
            x1 = x_new
            f1 = f_new

    return x1 if abs(f1) < abs(f2) else x2
