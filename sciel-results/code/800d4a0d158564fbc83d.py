import math
import time
import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin before 150s limit
    
    x = float(x0)
    fx = f(x)
    
    # Check if x0 is already a root
    if abs(fx) <= 1e-8:
        return x

    best_x = x
    best_abs = abs(fx)
    
    # Phase 1: Multi-scale scanning to find a bracket or good starting point
    # Use a set of step sizes to handle different frequency oscillations
    step_sizes = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    bracket_found = False
    lo = 0.0
    hi = 0.0
    
    for step in step_sizes:
        if time.time() > deadline:
            break
        
        # Scan in both directions with this step size
        for direction in [1, -1]:
            if time.time() > deadline:
                break
            
            # Do a limited scan (up to 10 steps in each direction per step size)
            for i in range(1, 11):
                if time.time() > deadline:
                    break
                
                x_test = x + direction * i * step
                try:
                    f_test = f(x_test)
                except:
                    continue
                
                if not math.isfinite(f_test):
                    continue
                
                abs_test = abs(f_test)
                if abs_test < best_abs:
                    best_abs = abs_test
                    best_x = x_test
                
                if abs_test <= 1e-8:
                    return x_test
                
                # Check for sign change with previous point
                # We need to track the last valid point
                if i == 1:
                    prev_x = x
                    prev_f = fx
                else:
                    prev_x = x + direction * (i - 1) * step
                    try:
                        prev_f = f(prev_x)
                    except:
                        continue
                    if not math.isfinite(prev_f):
                        continue
                
                if prev_f * f_test < 0:
                    lo, hi = min(prev_x, x_test), max(prev_x, x_test)
                    bracket_found = True
                    break
            
            if bracket_found:
                break
        
        if bracket_found:
            break

    # Phase 2: If bracket found, use Brent's method
    if bracket_found:
        try:
            root = brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=100)
            if abs(f(root)) <= 1e-8:
                return root
        except Exception:
            pass

    # Phase 3: If no bracket found, use Nelder-Mead on |f(x)|
    # This is a structural change: derivative-free global optimizer that handles oscillatory landscapes
    if best_abs > 1e-8:
        def obj(x):
            try:
                val = f(x)
                if not math.isfinite(val):
                    return 1e100
                return abs(val)
            except:
                return 1e100
        
        # Start Nelder-Mead from best_x
        try:
            # Create initial simplex around best_x
            scale = max(1.0, abs(best_x)) * 0.1
            x0_arr = np.array([best_x])
            
            res = minimize(obj, x0_arr, method='Nelder-Mead',
                          options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 200, 'adaptive': True})
            
            if res.fun <= 1e-8:
                return float(res.x[0])
            
            # Update best if improved
            if res.fun < best_abs:
                best_x = float(res.x[0])
                best_abs = res.fun
        except Exception:
            pass
        
        # Phase 4: Local refinement with minimize_scalar if still not good enough
        if best_abs > 1e-8:
            try:
                search_width = max(1.0, abs(best_x) * 0.5)
                res2 = minimize_scalar(obj, bounds=(best_x - search_width, best_x + search_width),
                                      method='bounded', options={'xatol': 1e-12, 'maxiter': 100})
                if res2.fun <= 1e-8:
                    return res2.x
                if res2.fun < best_abs:
                    best_x = res2.x
                    best_abs = res2.fun
            except Exception:
                pass
    
    return best_x
# EVOLVE-BLOCK-END
