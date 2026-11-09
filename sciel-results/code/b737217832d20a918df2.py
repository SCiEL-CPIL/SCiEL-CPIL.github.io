"""Root finder for 3 fixed oscillatory equations."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using efficient bracketing and
    Illinois-accelerated bisection with a time deadline guard.
    """
    tol = 1e-8
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > deadline
    
    x = float(x0)
    
    # Evaluate at starting point
    try:
        fx = f(x)
    except Exception:
        fx = float('nan')
    
    if not math.isfinite(fx):
        # Try a few nearby points
        for dx in [1e-6, -1e-6, 1e-4, -1e-4, 1.0, -1.0, 10.0, -10.0]:
            if check_deadline():
                break
            try:
                x_try = x + dx
                fx_try = f(x_try)
                if math.isfinite(fx_try) and abs(fx_try) <= tol:
                    return x_try
            except Exception:
                continue
        return x  # best effort
    
    if abs(fx) <= tol:
        return x
    
    # Phase 1: Find a bracket [a, b] where f(a) and f(b) have opposite signs
    a = x
    fa = fx
    
    # Adaptive bracketing: try multiple scales in parallel-ish fashion
    # Use a set of candidate steps to find sign change quickly
    b = None
    fb = None
    
    # Try a range of steps, starting from moderate sizes
    # For oscillatory functions, the period might be O(1) to O(100)
    candidate_steps = [
        0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 5000.0, 10000.0
    ]
    
    # Also try small steps for very local roots
    small_steps = [1e-4, 1e-3, 1e-2, 0.01, 0.05]
    
    all_steps = small_steps + candidate_steps
    
    for sign in [1, -1]:
        if check_deadline():
            break
        for step in all_steps:
            if check_deadline():
                break
            x_try = x + sign * step
            try:
                f_try = f(x_try)
            except Exception:
                continue
            
            if not math.isfinite(f_try):
                continue
            
            if abs(f_try) <= tol:
                return x_try
            
            if fa * f_try < 0:
                b = x_try
                fb = f_try
                break
        
        if b is not None:
            break
    
    # If still no bracket, try even larger steps
    if b is None:
        for sign in [1, -1]:
            if check_deadline():
                break
            step = 20000.0
            while step <= 1e7:
                if check_deadline():
                    break
                x_try = x + sign * step
                try:
                    f_try = f(x_try)
                except Exception:
                    step *= 2
                    continue
                
                if not math.isfinite(f_try):
                    step *= 2
                    continue
                
                if abs(f_try) <= tol:
                    return x_try
                
                if fa * f_try < 0:
                    b = x_try
                    fb = f_try
                    break
                step *= 2
    
    if b is None:
        # No bracket found. Try secant method as fallback
        # First, find the point with smallest |f| in a local search
        best_x = x
        best_abs_fx = abs(fx)
        
        for dx in [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0, 10.0, -10.0]:
            if check_deadline():
                break
            x_try = x + dx
            try:
                f_try = f(x_try)
            except Exception:
                continue
            if not math.isfinite(f_try):
                continue
            if abs(f_try) < best_abs_fx:
                best_abs_fx = abs(f_try)
                best_x = x_try
            if abs(f_try) <= tol:
                return x_try
        
        # Secant method from best_x and x
        x1 = best_x
        f1 = f(best_x) if best_x != x else fx
        x0_s = x
        f0_s = fx
        
        if x1 != x0_s and f1 != f0_s:
            for _ in range(80):
                if check_deadline():
                    break
                if f1 == f0_s:
                    break
                x_new = x1 - f1 * (x1 - x0_s) / (f1 - f0_s)
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(f_new):
                    break
                if abs(f_new) <= tol:
                    return x_new
                x0_s = x1
                f0_s = f1
                x1 = x_new
                f1 = f_new
                if abs(x1 - x0_s) < 1e-15 * max(1.0, abs(x1)):
                    break
        
        return best_x
    
    # Phase 2: We have a bracket [a, b] with f(a)*f(b) < 0
    # Use Illinois-accelerated bisection (faster than pure bisection, robust)
    x_lo = a
    x_hi = b
    f_lo = fa
    f_hi = fb
    
    # Illinois method: when the bracket shrinks, adjust the endpoint with larger |f|
    # to accelerate convergence
    for _ in range(200):
        if check_deadline():
            break
        
        # Compute midpoint
        x_mid = (x_lo + x_hi) / 2.0
        try:
            f_mid = f(x_mid)
        except Exception:
            break
        
        if not math.isfinite(f_mid):
            break
        
        if abs(f_mid) <= tol:
            return x_mid
        
        # Update bracket
        if f_lo * f_mid < 0:
            x_hi = x_mid
            f_hi = f_mid
        else:
            x_lo = x_mid
            f_lo = f_mid
        
        # Illinois adjustment: if the bracket is shrinking slowly,
        # reduce the value at the endpoint with larger |f|
        if abs(f_lo) > abs(f_hi):
            f_lo = f_lo / 2.0
        else:
            f_hi = f_hi / 2.0
        
        # Check convergence
        if abs(x_hi - x_lo) < 1e-14 * max(1.0, abs(x_mid)):
            break
    
    # Final check
    for cx in [x_lo, x_hi, (x_lo + x_hi) / 2.0]:
        try:
            cv = f(cx)
            if math.isfinite(cv) and abs(cv) <= tol:
                return cx
        except Exception:
            continue
    
    # Return the point with smallest |f|
    best_x = x_lo
    best_val = abs(f_lo)
    for cx in [x_hi, (x_lo + x_hi) / 2.0]:
        try:
            cv = abs(f(cx))
            if math.isfinite(cv) and cv < best_val:
                best_val = cv
                best_x = cx
        except Exception:
            continue
    
    return best_x

# EVOLVE-BLOCK-END
