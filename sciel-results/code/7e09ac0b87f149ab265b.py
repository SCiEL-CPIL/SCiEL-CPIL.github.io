import math
import time
import numpy as np
from scipy.optimize import brentq, minimize_scalar

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 490  # Leave margin for safety
    START_TIME = time.time()
    DEADLINE = START_TIME + 110.0  # 120s limit, 10s margin
    
    calls = 0
    best_x = float(x0)
    best_f = float('inf')
    
    # Cache to avoid redundant calls
    f_cache = {}
    
    def safe_f(x):
        nonlocal calls, best_x, best_f
        if calls >= MAX_CALLS:
            return float('inf')
        
        # Time check
        if time.time() > DEADLINE:
            return float('inf')
            
        # Use rounding for cache key to handle floating point noise
        key = round(float(x), 14)
        
        if key in f_cache:
            val = f_cache[key]
            # Update best even from cache
            if abs(val) < best_f:
                best_f = abs(val)
                best_x = float(x)
            return val

        calls += 1
        try:
            val = f(float(x))
            if not math.isfinite(val):
                f_cache[key] = float('inf')
                return float('inf')
            f_cache[key] = val
        except Exception:
            f_cache[key] = float('inf')
            return float('inf')
        
        if abs(val) < best_f:
            best_f = abs(val)
            best_x = float(x)
            
        return val

    x_start = float(x0)
    
    # 1. Check starting point
    fx0 = safe_f(x_start)
    if abs(fx0) <= TOL:
        return x_start

    # 2. Local Minimization Strategy (Structural Change)
    # Instead of scanning for sign changes (which can be expensive if roots are far or oscillation is high),
    # we directly minimize g(x) = f(x)^2 using a robust 1D optimizer.
    # This works for both simple roots and tangent roots (double roots) where sign change fails.
    # We start with a very local search around x0, then expand the range if it fails.
    
    # Objective function for minimization
    def objective(x):
        fx = safe_f(x)
        if not math.isfinite(fx):
            return float('inf')
        return fx * fx
    
    # Try a sequence of bounding boxes for the local minimizer
    # The idea is to find a root in a local neighborhood first.
    # If the function is oscillatory, a root is likely nearby unless x0 is far away.
    # We expand the search window if the local minimum is not a root.
    
    # Scale based on x0 to handle different magnitudes
    scale = max(1.0, abs(x_start))
    
    # Search ranges: [x0 - R, x0 + R]
    # Start small, grow exponentially
    ranges = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    
    found_root = False
    
    for R in ranges:
        if calls > MAX_CALLS - 20:
            break
        if time.time() > DEADLINE - 5.0:
            break
            
        lower = x_start - R * scale
        upper = x_start + R * scale
        
        # Use minimize_scalar with bounded method
        # This is robust and doesn't require sign changes.
        # maxiter is kept low to save calls for other attempts if needed.
        try:
            res = minimize_scalar(
                objective, 
                bounds=(lower, upper), 
                method='bounded',
                options={'maxiter': 50}
            )
            
            # Check if the result is a valid root
            if res.success:
                x_res = float(res.x)
                f_res = safe_f(x_res)
                
                if math.isfinite(x_res) and abs(f_res) <= TOL:
                    return x_res
                
                # If we found a very small value but not quite TOL, keep it as best
                # But continue searching in larger ranges just in case?
                # Actually, if we found a local min of f^2, it's likely the closest root in that region.
                # If it's not < TOL, it might be a local min that is not a root (e.g. f(x) = x^2 + 1e-4).
                # In that case, we should expand to find a true root if one exists nearby.
                
        except Exception:
            pass
            
        # If we haven't found a root yet, we continue to next range.
        # Note: best_x/best_f are updated inside safe_f, so we always have a candidate.

    # 3. Fallback: If local minimization didn't yield a root within TOL,
    # try a broader sign-change scan as a last resort, or just return best_x.
    # Given the budget, returning best_x is safer than risking more calls.
    # However, if best_f is still large, we might have missed the root entirely.
    # Let's do one final, sparse sign-change scan if we have budget.
    
    if best_f > TOL and calls < MAX_CALLS - 50:
        # Sparse scan for sign changes
        # Use a wider range
        R = 1000.0 * scale
        lower = x_start - R
        upper = x_start + R
        
        # Generate points
        num_pts = 50
        points = np.linspace(lower, upper, num_pts)
        
        vals = []
        for p in points:
            if time.time() > DEADLINE - 2.0:
                break
            if calls > MAX_CALLS - 5:
                break
            fp = safe_f(p)
            vals.append((p, fp))
            
        if len(vals) >= 2:
            for i in range(len(vals) - 1):
                x1, f1 = vals[i]
                x2, f2 = vals[i+1]
                
                if abs(f1) <= TOL:
                    return x1
                if abs(f2) <= TOL:
                    return x2
                    
                if f1 * f2 < 0:
                    # Bracket found, try brentq
                    try:
                        x_root = brentq(safe_f, x1, x2, xtol=1e-12, rtol=1e-12, maxiter=50)
                        f_root = safe_f(x_root)
                        if math.isfinite(x_root) and abs(f_root) <= TOL:
                            return x_root
                    except Exception:
                        pass

    return best_x
