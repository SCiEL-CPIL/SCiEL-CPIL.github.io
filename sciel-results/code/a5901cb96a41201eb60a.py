import numpy as np
from scipy.optimize import brentq, minimize_scalar

def find_root(f, x0):
    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Strategy 1: Find a bracket and use Brent's method
    a = x
    fa = fx
    
    # Start with a small step
    h = 1e-5
    # Expand up to 1e5
    max_h = 1e5
    
    while h <= max_h:
        # Check positive direction
        x_pos = x + h
        fx_pos = f(x_pos)
        if abs(fx_pos) <= 1e-8:
            return x_pos
        if fa * fx_pos < 0:
            try:
                root = brentq(f, x, x_pos, xtol=1e-12, rtol=1e-12, maxiter=100)
                return float(root)
            except Exception:
                pass
        
        # Check negative direction
        x_neg = x - h
        fx_neg = f(x_neg)
        if abs(fx_neg) <= 1e-8:
            return x_neg
        if fa * fx_neg < 0:
            try:
                root = brentq(f, x_neg, x, xtol=1e-12, rtol=1e-12, maxiter=100)
                return float(root)
            except Exception:
                pass
        
        # Expand step size
        h *= 10.0

    # Strategy 2: Fallback to local minimization of |f(x)|
    # This handles cases where bracketing failed (e.g., tangent root, or missed sign change)
    # Define a helper function for minimization
    def obj(z):
        return abs(f(z))
    
    # Search in a reasonable local window
    # If h expanded to 1e5, the root might be far, but x0 is a "starting point", so likely close.
    # Let's search in [x-1, x+1] first, then [x-10, x+10] if needed?
    # Minimize_scalar is bounded.
    try:
        res = minimize_scalar(obj, bounds=(x - 1.0, x + 1.0), method='bounded', options={'xatol': 1e-12})
        if res.success:
            x_min = res.x
            f_min = obj(x_min)
            if f_min <= 1e-8:
                return float(x_min)
    except Exception:
        pass
        
    try:
        res = minimize_scalar(obj, bounds=(x - 10.0, x + 10.0), method='bounded', options={'xatol': 1e-12})
        if res.success:
            x_min = res.x
            f_min = obj(x_min)
            if f_min <= 1e-8:
                return float(x_min)
    except Exception:
        pass

    # Final fallback: return the point with the smallest |f(x)| found during the process
    # We didn't store all points, so just return x0 or the last evaluated point?
    # Better to return x0 as it's the "best" we know without extra calls? 
    # Or we can do a few random samples? No, deterministic is better.
    # If we get here, we likely fail the equation.
    return x
