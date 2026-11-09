"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import numpy as np
from scipy.optimize import minimize_scalar

def find_root(f, x0):
    """
    Find a root of f near x0 using global optimization of f(x)^2.
    """
    x0 = float(x0)
    
    # 1. Check if x0 is already a root
    try:
        fx0 = f(x0)
        if math.isfinite(fx0) and abs(fx0) <= 1e-8:
            return x0
    except Exception:
        pass

    def objective(x):
        try:
            fx = f(x)
            if not math.isfinite(fx):
                return 1e20
            return fx * fx
        except Exception:
            return 1e20

    # 2. Try minimize_scalar with expanding bounds
    # We try a few scales to ensure we cover the likely root location
    scales = [10.0, 100.0, 1000.0]
    
    best_x = None
    best_val = float('inf')
    
    for scale in scales:
        lower = x0 - scale
        upper = x0 + scale
        
        try:
            res = minimize_scalar(objective, bounds=(lower, upper), method='bounded', options={'xatol': 1e-12, 'maxiter': 100})
            
            if res.success:
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_x = res.x
                    
                # Check if we found a root
                if val <= 1e-16: # |f(x)| <= 1e-8
                    # Polish with a few Newton steps if possible to ensure precision
                    x_polished = res.x
                    for _ in range(5):
                        try:
                            fx = f(x_polished)
                            if abs(fx) <= 1e-12:
                                break
                            h = 1e-8 * max(1.0, abs(x_polished))
                            dfx = (f(x_polished + h) - f(x_polished - h)) / (2 * h)
                            if abs(dfx) < 1e-15:
                                break
                            x_new = x_polished - fx / dfx
                            if not math.isfinite(x_new):
                                break
                            x_polished = x_new
                        except Exception:
                            break
                    
                    if math.isfinite(x_polished):
                        try:
                            if abs(f(x_polished)) <= 1e-8:
                                return x_polished
                        except Exception:
                            pass
                            
                    return res.x
        except Exception:
            continue

    # 3. If minimization didn't find a root within threshold, return best candidate
    if best_x is not None and math.isfinite(best_x):
        return best_x
        
    return x0
# EVOLVE-BLOCK-END
