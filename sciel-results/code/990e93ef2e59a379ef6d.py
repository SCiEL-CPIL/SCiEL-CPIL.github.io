"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import numpy as np

def find_root(f, x0):
    """
    Find a root of f near x0 using a robust combination of methods:
    1. Try Newton's method with finite-difference derivative (fast for smooth polynomials)
    2. If Newton fails or diverges, use Brent's method with a bracket found by expanding from x0
    3. Use scipy.optimize.minimize_scalar as a fallback for global minimization of f(x)^2
    
    Returns x with |f(x)| <= 1e-8.
    """
    x0 = float(x0)
    tol = 1e-12
    max_newton = 50
    
    # First, try to see if x0 is already a root
    try:
        fx0 = f(x0)
        if not math.isfinite(fx0):
            raise ValueError("non-finite")
        if abs(fx0) <= 1e-8:
            return x0
    except Exception:
        pass
    
    # Method 1: Newton's method with finite-difference derivative
    x = x0
    converged = False
    for _ in range(max_newton):
        try:
            fx = f(x)
            if not math.isfinite(fx):
                break
            if abs(fx) <= 1e-8:
                return x
            h = 1e-7 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if not math.isfinite(d) or abs(d) < 1e-15:
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            # Damping if step is too large
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x + 10 * max(1.0, abs(x)) * (1 if x_new > x else -1)
            x = x_new
        except Exception:
            break
    
    if converged or (abs(f(x)) <= 1e-8 if math.isfinite(f(x)) else False):
        return x
    
    # Method 2: Find a bracket and use Brent's method
    # Expand around x0 to find a sign change
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
            # Expand
            for _ in range(30):
                a = x0 - 2 * (b - x0)
                b = x0 + 2 * (b - x0)
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
    
    if bracket_found:
        # Use Brent's method via scipy
        from scipy.optimize import brentq
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-14, maxiter=100)
            if math.isfinite(root) and abs(f(root)) <= 1e-8:
                return root
        except Exception:
            pass
    
    # Method 3: minimize_scalar on f(x)^2
    from scipy.optimize import minimize_scalar
    
    def objective(x):
        try:
            fx = f(x)
            if not math.isfinite(fx):
                return 1e20
            return fx * fx
        except Exception:
            return 1e20
    
    # Try several starting points for the minimization
    best_x = None
    best_val = float('inf')
    
    search_range = max(1.0, abs(x0))
    for start_offset in [0, -search_range, search_range, -2*search_range, 2*search_range]:
        x_start = x0 + start_offset
        try:
            result = minimize_scalar(objective, bounds=(x_start - 10*search_range, x_start + 10*search_range), 
                                      method='bounded', options={'xatol': 1e-12, 'maxiter': 200})
            if result.success and math.isfinite(result.fun) and result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            continue
    
    if best_x is not None and math.isfinite(best_x):
        try:
            if abs(f(best_x)) <= 1e-8:
                return best_x
        except Exception:
            pass
    
    # Fallback: return the best candidate we have
    if best_x is not None and math.isfinite(best_x):
        return best_x
    
    return x0

# EVOLVE-BLOCK-END
