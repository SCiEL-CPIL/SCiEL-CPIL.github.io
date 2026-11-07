import numpy as np
from scipy.optimize import brentq, minimize_scalar

def find_root(f, x0):
    x0 = float(x0)
    
    # First, check if x0 is already a root
    try:
        fx0 = f(x0)
        if np.isfinite(fx0) and abs(fx0) <= 1e-8:
            return x0
    except:
        pass

    # Strategy: Hierarchical Grid Search to find a bracket [a, b]
    # where f(a) and f(b) have opposite signs.
    # We use ranges that expand outwards.
    # Cost per level: ~100 calls.
    # Total budget: ~400 calls for search + ~20 for Brent = 420. Safe.
    
    ranges = [
        (x0 - 1.0, x0 + 1.0),
        (x0 - 10.0, x0 + 10.0),
        (x0 - 100.0, x0 + 100.0),
        (x0 - 1000.0, x0 + 1000.0),
    ]
    
    # We also want to cover [0, 1] and [-1, 0] specifically if x0 is far away,
    # but the ranges above centered on x0 might miss roots far from x0.
    # Let's add fixed global ranges.
    global_ranges = [
        (-1.0, 1.0),
        (-10.0, 10.0),
        (-100.0, 100.0),
    ]
    
    all_ranges = global_ranges + ranges
    
    best_x = x0
    best_val = float('inf')
    
    for a, b in all_ranges:
        # Skip if range is invalid
        if a >= b:
            continue
            
        # Number of points: 50 points per half-range? 
        # Let's use 100 points total per range.
        n_points = 100
        x_vals = np.linspace(a, b, n_points)
        
        prev_x = None
        prev_f = None
        
        for x in x_vals:
            try:
                fx = f(x)
                if not np.isfinite(fx):
                    prev_x, prev_f = None, None
                    continue
                    
                # Update best candidate for fallback
                if abs(fx) < best_val:
                    best_val = abs(fx)
                    best_x = x
                    
                # Check for sign change
                if prev_f is not None:
                    if prev_f * fx < 0:
                        # Found bracket [prev_x, x]
                        try:
                            root = brentq(f, prev_x, x, xtol=1e-12, rtol=1e-12, maxiter=50)
                            # Verify root
                            f_root = f(root)
                            if np.isfinite(f_root) and abs(f_root) <= 1e-8:
                                return root
                            else:
                                # Brent might have converged to a non-root or failed
                                # Return the best of root or endpoints
                                if abs(f_root) < best_val:
                                    best_val = abs(f_root)
                                    best_x = root
                        except Exception:
                            pass
                    
                prev_x, prev_f = x, fx
                
            except Exception:
                prev_x, prev_f = None, None
                
        # Early exit if we found a very good root? 
        # No, we need to ensure |f(x)| <= 1e-8.
        # If best_val is already small, we can stop searching further ranges?
        # But we might find an exact zero later. However, if best_val < 1e-8, we can return.
        if best_val <= 1e-8:
            return best_x

    # If no bracket found, return the best point found
    if best_val < float('inf'):
        return best_x
        
    # Fallback: Try local minimization around x0
    try:
        res = minimize_scalar(lambda x: abs(f(x)), bounds=(x0 - 1.0, x0 + 1.0), method='bounded', options={'xatol': 1e-12})
        if res.success and np.isfinite(res.fun) and res.fun <= 1e-8:
            return res.x
        if np.isfinite(res.fun) and res.fun < best_val:
            return res.x
    except:
        pass
        
    return best_x
