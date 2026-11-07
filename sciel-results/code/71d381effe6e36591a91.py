"""
Strategy: Dense Local Adaptive Grid with Brent's Method Refinement
1. Local Dense Search: Perform a linear sweep of ~200 points in a window around x0.
   The window size is adaptive: starts small (1e-2 * scale) and expands if no sign change is found.
   This is much more likely to find the "intended" root near x0 than a global log-sweep, 
   and provides many tight brackets to choose from.
2. Optimal Refinement: Select the bracket with the smallest width and use scipy.optimize.brentq.
   Brent's method is guaranteed to converge (like bisection) but much faster (superlinear), 
   typically requiring < 20 calls to reach 1e-12 precision.
3. Budget & Time Management: 
   - Strict call limit (300) to keep score high.
   - Time limit guard (115s) to prevent timeout.
   - If no bracket is found in local windows, fallback to a broader log-sweep.
"""
import math
import time
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 115.0  # Safety margin for 120s limit
    MAX_CALLS = 300     # Keep well below 500 to maximize score component
    TOL = 1e-10         # Target tolerance for brentq
    
    calls = 0
    cache = {}
    
    def _f(x):
        nonlocal calls
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError("Time limit exceeded")
        
        # Check cache
        key = x
        if key in cache:
            return cache[key]
            
        if calls >= MAX_CALLS:
            raise RuntimeError("Call limit exceeded")
            
        calls += 1
        val = f(x)
        cache[key] = val
        return val

    # State for best result found so far
    best_x = float('nan')
    best_f_val = float('inf')
    
    def update_best(x, fx):
        nonlocal best_x, best_f_val
        if math.isfinite(x) and math.isfinite(fx):
            if abs(fx) < abs(best_f_val):
                best_x = x
                best_f_val = fx

    # 1. Initial check
    try:
        fx0 = _f(x0)
        if abs(fx0) <= TOL:
            return x0
        update_best(x0, fx0)
    except Exception:
        return x0

    # 2. Local Dense Search for a Tight Bracket
    # We try several window sizes around x0, starting small.
    # scale = max(1.0, abs(x0))
    # Windows: 1e-3, 1e-2, 1e-1, 1, 10, 100 (multiplied by scale)
    scale = max(1.0, abs(x0))
    window_factors = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]
    
    bracket = None
    best_bracket_width = float('inf')
    
    for w_factor in window_factors:
        if calls > MAX_CALLS - 50: # Leave budget for refinement
            break
            
        win_size = w_factor * scale
        # If x0 is very small, ensure window is not zero
        if win_size < 1e-6:
            win_size = 1e-6
            
        a = x0 - win_size
        b = x0 + win_size
        
        # We want to find a sign change. 
        # Use a moderate number of points, e.g., 50-100.
        # As window expands, we can use fewer points or more? 
        # Let's use 50 points for small windows, 20 for large ones to save calls.
        n_points = 50 if w_factor < 1.0 else 20
        
        if calls + n_points > MAX_CALLS - 20:
            n_points = max(5, MAX_CALLS - 20 - calls)
            
        if n_points <= 1:
            continue
            
        try:
            xs = np.linspace(a, b, n_points)
            # Evaluate f at these points
            prev_x = None
            prev_fx = None
            local_bracket = None
            local_width = float('inf')
            
            for x in xs:
                # Skip if we already have this point in cache? 
                # _f handles caching.
                try:
                    fx = _f(x)
                    update_best(x, fx)
                    
                    if prev_fx is not None:
                        if prev_fx * fx < 0:
                            # Found a bracket
                            width = abs(x - prev_x)
                            if width < local_width:
                                local_bracket = (prev_x, x)
                                local_width = width
                except Exception:
                    # If f fails at a point, stop this window
                    break
                
                prev_x = x
                prev_fx = fx
            
            if local_bracket:
                if local_width < best_bracket_width:
                    bracket = local_bracket
                    best_bracket_width = local_width
                    
                # If we found a very tight bracket, we can stop searching larger windows
                # A width of < 1e-4 is likely sufficient for fast Brent convergence
                if best_bracket_width < 1e-4 * scale:
                    break
                    
        except Exception:
            continue

    # 3. Fallback: Global Logarithmic Sweep if local search failed
    if not bracket:
        # Try a few logarithmic steps outward
        base_offsets = [1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
        points = [(x0, fx0)]
        
        for s in base_offsets:
            if calls > MAX_CALLS - 20:
                break
            xp = x0 + s * scale
            xn = x0 - s * scale
            
            try:
                fxp = _f(xp)
                update_best(xp, fxp)
                for (xq, fxq) in points:
                    if fxp * fxq < 0:
                        bracket = (xp, xq)
                        break
                points.append((xp, fxp))
            except Exception:
                pass
            
            if bracket:
                break
                
            if calls > MAX_CALLS - 20:
                break
                
            try:
                fxn = _f(xn)
                update_best(xn, fxn)
                for (xq, fxq) in points:
                    if fxn * fxq < 0:
                        bracket = (xn, xq)
                        break
                points.append((xn, fxn))
            except Exception:
                pass
            
            if bracket:
                break

    # 4. Refinement using Brent's Method
    if bracket:
        a, b = bracket
        try:
            fa = _f(a)
            fb = _f(b)
        except Exception:
            fa, fb = None, None
            
        if fa is not None and fb is not None and fa * fb < 0:
            # Use brentq for fast convergence
            # maxiter=50 is plenty for 1e-10 tolerance
            try:
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
                # Verify
                if math.isfinite(root):
                    f_root = _f(root)
                    if abs(f_root) <= 1e-8:
                        return root
                    else:
                        # Fallback to best found
                        if math.isfinite(best_x):
                            return best_x
                        return root
            except Exception:
                # If brentq fails, try Illinois manually as fallback
                # Or just return best_x
                if math.isfinite(best_x):
                    return best_x
                return (a+b)/2.0

    # If no bracket found, return best_x
    if math.isfinite(best_x):
        return best_x
    return x0

# EVOLVE-BLOCK-END
