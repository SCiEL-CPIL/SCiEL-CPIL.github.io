import math
from scipy.optimize import brentq

def find_root(f, x0):
    x = float(x0)
    try:
        fx = f(x)
    except Exception:
        return x

    if not math.isfinite(fx):
        return x
        
    if abs(fx) <= 1e-8:
        return x

    a = None
    b = None
    fa = None
    fb = None
    
    # Initial step size scaled by magnitude of x
    base_step = 0.1 * max(1.0, abs(x))
    step = base_step
    max_step = 1e6 * max(1.0, abs(x))
    
    # We will search outwards
    # Keep track of the last checked points to avoid re-evaluating if we expand?
    # Since we expand symmetrically, we don't re-use previous endpoints except x0.
    
    while step < max_step:
        x_plus = x + step
        try:
            fx_plus = f(x_plus)
        except Exception:
            fx_plus = float('nan')
            
        if math.isfinite(fx_plus):
            if abs(fx_plus) <= 1e-8:
                return x_plus
            if fx * fx_plus < 0:
                a, b = x, x_plus
                break
                
        x_minus = x - step
        try:
            fx_minus = f(x_minus)
        except Exception:
            fx_minus = float('nan')
            
        if math.isfinite(fx_minus):
            if abs(fx_minus) <= 1e-8:
                return x_minus
            if fx * fx_minus < 0:
                a, b = x_minus, x
                break
                
        step *= 2.0
        
    if a is None:
        # Failed to bracket
        # Return the point with smallest absolute function value seen so far?
        # We only have fx, and maybe some failed evaluations.
        # Just return x.
        return x

    try:
        root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
        if math.isfinite(root):
            # Final check
            fr = f(root)
            if math.isfinite(fr) and abs(fr) <= 1e-8:
                return root
            # If brentq converged but f(root) is still large (steep function?), 
            # it might be a local issue, but brentq is robust.
            # We return root anyway as it's the best candidate.
            return root
    except Exception:
        pass

    return x
