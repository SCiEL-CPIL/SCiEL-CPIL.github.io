"""Root finder for 3 fixed oscillatory equations."""
import math
import time

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 for a given function f and starting point x0.
    Uses a combination of:
      1. Multi-scale secant seeding (structural idea from inspiration 1)
      2. Cached function evaluation with call counting (from inspiration 2)
      3. Adaptive secant iteration with best-tracking
      4. Newton polish for final convergence
      5. Deadline guard to avoid timeouts
    """
    tol = 1e-8
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    MAX_CALLS = 495
    
    x0 = float(x0)
    calls = 0
    cache = {}
    best_x = x0
    best_f = float('inf')
    found = False
    
    def safe_f(x):
        nonlocal calls, best_x, best_f, found
        if found:
            return 0.0
        if calls >= MAX_CALLS:
            return float('nan')
        if time.time() - start_time > deadline:
            return float('nan')
        
        if x in cache:
            val = cache[x]
        else:
            calls += 1
            try:
                val = f(x)
                val = float(val)
            except Exception:
                val = float('nan')
            cache[x] = val
            
        if math.isfinite(val):
            abs_val = abs(val)
            if abs_val < best_f:
                best_f = abs_val
                best_x = x
                if best_f <= tol:
                    found = True
        return val
    
    # Initial evaluation
    fx0 = safe_f(x0)
    if found:
        return best_x
    
    if not math.isfinite(fx0):
        # Try a few nearby points
        for h in [1e-4, 1e-3, 1e-2, 0.1, 1.0]:
            x_try = x0 + h
            fx_try = safe_f(x_try)
            if found:
                return best_x
            if math.isfinite(fx_try):
                break
        return best_x
    
    # Phase 1: Multi-scale secant seeding
    # Generate secant pairs at different scales to catch oscillations
    scales = [1e-2, 1e-1, 1.0, 1e-3, 1e-4, 10.0, 1e-5, 100.0, 1000.0]
    
    for scale in scales:
        if found:
            return best_x
        if time.time() - start_time > deadline:
            break
        
        # Try multiple pair configurations at this scale
        # Config 1: (x0, x0+scale)
        # Config 2: (x0-scale, x0+scale)
        # Config 3: (x0+scale, x0+2*scale)
        
        configs = [
            (0, 1),    # (x0, x0+scale)
            (-1, 1),   # (x0-scale, x0+scale)
            (1, 2),    # (x0+scale, x0+2*scale)
        ]
        
        for sign_a, sign_b in configs:
            if found:
                return best_x
            if time.time() - start_time > deadline:
                break
            
            x_a = x0 + scale * sign_a
            x_b = x0 + scale * sign_b
            
            if abs(x_a - x_b) < 1e-12 * max(1.0, abs(x0)):
                continue
            
            f_a = safe_f(x_a)
            if found:
                return best_x
            f_b = safe_f(x_b)
            if found:
                return best_x
            
            if not math.isfinite(f_a) or not math.isfinite(f_b):
                continue
            
            # Check for immediate success
            if abs(f_a) <= tol:
                return x_a
            if abs(f_b) <= tol:
                return x_b
            
            # Run secant method from this pair
            x_curr_a = x_a
            x_curr_b = x_b
            f_curr_a = f_a
            f_curr_b = f_b
            
            for _ in range(35):
                if found:
                    return best_x
                if time.time() - start_time > deadline:
                    break
                
                if abs(f_curr_a) <= tol:
                    return x_curr_a
                if abs(f_curr_b) <= tol:
                    return x_curr_b
                
                denom = f_curr_b - f_curr_a
                if denom == 0 or not math.isfinite(denom):
                    break
                
                x_new = x_curr_b - f_curr_b * (x_curr_b - x_curr_a) / denom
                
                if not math.isfinite(x_new):
                    break
                
                # Shift window
                x_curr_a = x_curr_b
                f_curr_a = f_curr_b
                x_curr_b = x_new
                f_curr_b = safe_f(x_new)
                
                if found:
                    return best_x
                
                if not math.isfinite(f_curr_b):
                    break
                
                # Check convergence
                if abs(f_curr_b) <= tol:
                    return x_curr_b
    
    # Phase 2: Newton polish on the best point found
    if not found and math.isfinite(best_f) and best_f < 1.0:
        x = best_x
        for _ in range(15):
            if found:
                return best_x
            if time.time() - start_time > deadline:
                break
            
            fx = safe_f(x)
            if found:
                return best_x
            if not math.isfinite(fx):
                break
            
            if abs(fx) <= tol:
                return x
            
            # Finite difference derivative
            h = 1e-6 * max(1.0, abs(x))
            fx2 = safe_f(x + h)
            if found:
                return best_x
            if not math.isfinite(fx2):
                break
            
            d = (fx2 - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            
            # Newton step
            x_new = x - fx / d
            
            if not math.isfinite(x_new):
                break
            
            # Damping: if step is too large, shrink it
            max_step = 10.0 * max(1.0, abs(x))
            step = x_new - x
            if abs(step) > max_step:
                x_new = x + math.copysign(max_step, step)
            
            x = x_new
    
    return best_x
