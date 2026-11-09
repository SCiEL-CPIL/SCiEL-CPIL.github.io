"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import numpy as np

def find_root(f, x0):
    """
    Find a root of f near x0 using a robust combination of methods:
    1. Newton's method with finite-difference derivative
    2. Bracket expansion and Brent's method
    3. Bounded scalar minimization of f(x)^2 as a last resort
    """
    x0 = float(x0)
    tol = 1e-12
    max_newton = 20
    
    # 1. Check if x0 is already a root
    try:
        fx0 = f(x0)
        if not math.isfinite(fx0):
            raise ValueError("non-finite")
        if abs(fx0) <= 1e-8:
            return x0
    except Exception:
        pass

    # 2. Newton's Method
    # We use a damped Newton step to prevent divergence.
    x = x0
    fx = None
    try:
        fx = f(x)
    except Exception:
        fx = None
        
    if fx is not None and math.isfinite(fx) and abs(fx) <= 1e-8:
        return x

    for _ in range(max_newton):
        if fx is None or not math.isfinite(fx):
            break
            
        # Finite difference derivative
        h = 1e-6 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
            if not math.isfinite(fx_plus):
                break
            d = (fx_plus - fx) / h
            if not math.isfinite(d) or abs(d) < 1e-15:
                break
        except Exception:
            break
            
        # Newton step
        step = fx / d
        x_new = x - step
        
        # Damping: if step is too large, limit it
        if not math.isfinite(x_new):
            break
            
        # Heuristic: if the step is very large, damp it
        max_step = 10.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            x_new = x + max_step * (1 if step > 0 else -1)
            
        x = x_new
        try:
            fx = f(x)
        except Exception:
            fx = None
            
        if fx is not None and math.isfinite(fx) and abs(fx) <= 1e-8:
            return x

    # 3. Bracketing and Brent's Method
    # Try to find a bracket [a, b] such that f(a) * f(b) < 0
    # Start with a small interval around x0 and expand
    a = x0 - 1.0
    b = x0 + 1.0
    fa = None
    fb = None
    
    try:
        fa = f(a)
        fb = f(b)
    except Exception:
        pass
    
    bracket_found = False
    if fa is not None and fb is not None and math.isfinite(fa) and math.isfinite(fb):
        if fa * fb <= 0:
            bracket_found = True
        else:
            # Expand the interval
            current_width = 2.0
            for _ in range(20):
                a = x0 - current_width
                b = x0 + current_width
                try:
                    fa = f(a)
                    fb = f(b)
                except Exception:
                    break
                if not math.isfinite(fa) or not math.isfinite(fb):
                    break
                if fa * fb <= 0:
                    bracket_found = True
                    break
                current_width *= 2.0
                if current_width > 1e6:
                    break

    if bracket_found:
        from scipy.optimize import brentq
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-14, maxiter=50)
            if math.isfinite(root):
                # Verify
                try:
                    if abs(f(root)) <= 1e-8:
                        return root
                except Exception:
                    pass
        except Exception:
            pass

    # 4. Fallback: Minimize f(x)^2 in a bounded region
    # Use a reasonable bound around x0
    from scipy.optimize import minimize_scalar
    
    def objective(x):
        try:
            fx = f(x)
            if not math.isfinite(fx):
                return 1e20
            return fx * fx
        except Exception:
            return 1e20

    # Search in a window around x0
    # If x0 is large, the window should be larger, but not too large to waste calls
    window = max(10.0, 2.0 * abs(x0))
    lower = x0 - window
    upper = x0 + window
    
    try:
        result = minimize_scalar(objective, bounds=(lower, upper), 
                                  method='bounded', options={'xatol': 1e-12, 'maxiter': 100})
        if result.success and math.isfinite(result.fun):
            x_candidate = result.x
            # Verify
            try:
                if abs(f(x_candidate)) <= 1e-8:
                    return x_candidate
            except Exception:
                pass
            # If not a root, return it anyway as best effort? 
            # The task says "returns x with |f(x)| <= 1e-8". 
            # If we can't find one, we return the best we have.
            return x_candidate
    except Exception:
        pass

    # Last resort: return x0 or the last Newton iterate
    return x

# EVOLVE-BLOCK-END
