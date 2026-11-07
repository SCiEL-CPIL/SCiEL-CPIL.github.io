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
        
    if abs(fx) <= 1e-12:
        return x

    # Strategy 1: Find a bracket and use Brent's method
    # Try expanding around x0
    a, b = None, None
    fa = fx
    step = 1.0
    max_expansions = 20
    
    for _ in range(max_expansions):
        # Try x + step
        x_plus = x + step
        try:
            fx_plus = f(x_plus)
            if math.isfinite(fx_plus) and fa * fx_plus < 0:
                a, b = x, x_plus
                break
        except Exception:
            pass
            
        # Try x - step
        x_minus = x - step
        try:
            fx_minus = f(x_minus)
            if math.isfinite(fx_minus) and fa * fx_minus < 0:
                a, b = x_minus, x
                break
        except Exception:
            pass
            
        step *= 2.0

    if a is not None and b is not None:
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            if math.isfinite(root):
                return root
        except Exception:
            pass # Fall through to fallback

    # Strategy 2: Fallback to Robust Secant (Illinois Algorithm)
    # This doesn't require a bracket and is more robust than plain Newton for poor starts
    x1 = x
    x2 = x + 1e-6 * max(1.0, abs(x))
    
    try:
        f1 = fx
        f2 = f(x2)
    except Exception:
        return x
        
    if not math.isfinite(f2):
        return x

    # Illinois Algorithm
    for _ in range(100):
        if abs(f1) < abs(f2):
            # Swap to keep f1 as the "best" (smallest abs value)
            x1, x2 = x2, x1
            f1, f2 = f2, f1
            
        if abs(f1) <= 1e-12:
            return x1
            
        # Secant step
        if f2 == f1:
            # Degenerate case, perturb
            x2 = x1 + 1e-6
            try:
                f2 = f(x2)
            except Exception:
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
        # halve the step from x1 to x_new
        if (x_new - x1) * (x2 - x1) > 0:
             # It's on the same side
             # Standard Illinois: replace x2 with (x1+x2)/2 if f_new has same sign as f2
             if (f_new - f1) * (f2 - f1) > 0:
                 x2 = (x1 + x2) / 2.0
                 try:
                     f2 = f(x2)
                 except Exception:
                     return x1
                 if not math.isfinite(f2):
                     return x1
        else:
             # It's on the other side, standard secant update
             x2 = x1
             f2 = f1
             x1 = x_new
             f1 = f_new
             # Wait, the logic above swaps x1/x2 to keep f1 smallest.
             # Let's simplify: Just use standard Secant with damping.

    # Re-implementing a cleaner Secant/Illinois for clarity and correctness
    x1 = x
    x2 = x + 1e-6 * max(1.0, abs(x))
    try:
        f1 = f(x1)
        f2 = f(x2)
    except Exception:
        return x
        
    for _ in range(100):
        if abs(f1) <= 1e-12:
            return x1
        if abs(f2) <= 1e-12:
            return x2
            
        if f2 == f1:
            x2 = x1 + 1e-6
            try: f2 = f(x2)
            except: return x1
            continue

        x0_new = x1 - f1 * (x1 - x2) / (f1 - f2)
        
        if not math.isfinite(x0_new):
            x0_new = (x1 + x2) / 2.0
            
        try:
            f0_new = f(x0_new)
        except Exception:
            return x1
            
        if not math.isfinite(f0_new):
            return x1

        # Illinois rule: if x0_new is between x1 and x2, or on the other side?
        # Standard Illinois:
        # If (x0_new - x1) and (x2 - x1) have same sign, and (f0_new - f1) and (f2 - f1) have same sign:
        #   set x2 = (x1 + x2) / 2
        # Else:
        #   set x2 = x1, f2 = f1
        #   set x1 = x0_new, f1 = f0_new
        
        if (x0_new - x1) * (x2 - x1) > 0 and (f0_new - f1) * (f2 - f1) > 0:
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
            x1 = x0_new
            f1 = f0_new

    return x1 if abs(f1) < abs(f2) else x2
