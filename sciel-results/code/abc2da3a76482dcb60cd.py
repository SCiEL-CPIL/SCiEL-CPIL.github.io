"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import numpy as np
from scipy.optimize import minimize_scalar

def find_root(f, x0):
    """
    Find a root of f near x0 using global optimization of f(x)^2 followed by Newton refinement.
    This approach leverages the smoothness of polynomials to locate zeros via minimization,
    avoiding the pitfalls of manual bracketing.
    """
    x0 = float(x0)
    tol = 1e-8
    
    # Objective function: square of f(x) to find zeros
    def obj(x):
        try:
            fx = f(x)
            if not math.isfinite(fx):
                return 1e100
            return fx * fx
        except Exception:
            return 1e100

    # 1. Check if x0 is already a root
    try:
        fx0 = f(x0)
        if math.isfinite(fx0) and abs(fx0) <= tol:
            return x0
    except Exception:
        pass

    best_x = x0
    best_val = obj(x0)
    
    # 2. Global search via minimize_scalar on f(x)^2
    # We try a sequence of expanding bounds around x0
    # This helps locate the root if it is not immediately close to x0
    radii = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 10000.0]
    
    for r in radii:
        a = x0 - r
        b = x0 + r
        try:
            res = minimize_scalar(obj, bounds=(a, b), method='bounded', options={'xatol': 1e-12, 'maxiter': 100})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
                # If we found a very small value, we might have found the root
                if best_val < 1e-16:
                    break
        except Exception:
            continue

    # 3. Refine the best candidate using Newton's method
    # This ensures we meet the strict |f(x)| <= 1e-8 tolerance
    x = best_x
    try:
        fx = f(x)
        if math.isfinite(fx):
            for _ in range(20):
                if abs(fx) <= tol:
                    return x
                h = 1e-7 * max(1.0, abs(x))
                d = (f(x + h) - fx) / h
                if not math.isfinite(d) or abs(d) < 1e-15:
                    break
                x_new = x - fx / d
                if not math.isfinite(x_new):
                    break
                # Damping
                if abs(x_new - x) > 10 * max(1.0, abs(x)):
                    x_new = x + 10 * max(1.0, abs(x)) * (1 if x_new > x else -1)
                
                fx_new = f(x_new)
                if not math.isfinite(fx_new):
                    break
                if abs(fx_new) < abs(fx):
                    x = x_new
                    fx = fx_new
                else:
                    break
    except Exception:
        pass

    # 4. Final verification
    try:
        if math.isfinite(x) and abs(f(x)) <= tol:
            return x
    except Exception:
        pass

    # If all else fails, return the best minimization result
    return best_x

# EVOLVE-BLOCK-END
