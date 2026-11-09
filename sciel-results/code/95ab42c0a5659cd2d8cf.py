"""Root finder for oscillatory equations using bracketing + Newton refinement."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    call_count = 0
    best_x = x0
    best_fabs = float('inf')
    
    def _f(x):
        nonlocal call_count
        val = f(x)
        call_count += 1
        return val
    
    def _check_time():
        return (time.time() - start_time) > DEADLINE
    
    # Phase 1: Find a bracket by sampling around x0
    # Try multiple scales to find sign change
    base_h = 1e-3 * max(1.0, abs(x0))
    fx0 = _f(x0)
    if abs(fx0) <= 1e-8:
        return x0
    best_x = x0
    best_fabs = abs(fx0)
    
    if _check_time():
        return best_x
    
    # Try to find a bracket by expanding and also trying smaller steps
    bracket = None
    found_left = None
    found_right = None
    
    # First, try to find sign change in a small neighborhood
    for scale in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        h = base_h * scale
        if _check_time():
            break
        
        # Check x0 - h
        x_left = x0 - h
        fx_left = _f(x_left)
        if abs(fx_left) <= 1e-8:
            return x_left
        if abs(fx_left) < best_fabs:
            best_x = x_left
            best_fabs = abs(fx_left)
        
        if fx0 * fx_left < 0:
            bracket = (x_left, x0, fx_left, fx0)
            break
        
        # Check x0 + h
        x_right = x0 + h
        fx_right = _f(x_right)
        if abs(fx_right) <= 1e-8:
            return x_right
        if abs(fx_right) < best_fabs:
            best_x = x_right
            best_fabs = abs(fx_right)
        
        if fx0 * fx_right < 0:
            bracket = (x0, x_right, fx0, fx_right)
            break
    
    if _check_time():
        return best_x
    
    # If no bracket found, try a finer sweep in a wider range
    if bracket is None:
        # Try a sweep with more points
        sweep_h = base_h * 1024
        n_sweep = 20
        prev_x = x0 - sweep_h
        prev_fx = _f(prev_x)
        if abs(prev_fx) <= 1e-8:
            return prev_x
        if abs(prev_fx) < best_fabs:
            best_x = prev_x
            best_fabs = abs(prev_fx)
        
        for i in range(1, n_sweep + 1):
            if _check_time():
                break
            x_i = x0 - sweep_h + 2 * sweep_h * i / n_sweep
            fx_i = _f(x_i)
            if abs(fx_i) <= 1e-8:
                return x_i
            if abs(fx_i) < best_fabs:
                best_x = x_i
                best_fabs = abs(fx_i)
            
            if prev_fx * fx_i < 0:
                bracket = (prev_x, x_i, prev_fx, fx_i)
                break
            prev_x = x_i
            prev_fx = fx_i
    
    if _check_time():
        return best_x
    
    # Phase 2: Refine using bisection + Newton
    if bracket is not None:
        a, b, fa, fb = bracket
        # Ensure fa and fb have opposite signs
        if fa * fb > 0:
            a, b = b, a
            fa, fb = fb, fa
        
        # Bisection to narrow down
        for i in range(50):
            if _check_time():
                break
            mid = (a + b) / 2.0
            fmid = _f(mid)
            if abs(fmid) <= 1e-10:
                return mid
            if abs(fmid) < best_fabs:
                best_x = mid
                best_fabs = abs(fmid)
            
            if fa * fmid <= 0:
                b = mid
                fb = fmid
            else:
                a = mid
                fa = fmid
            
            # If interval is small enough, switch to Newton
            if abs(b - a) < 1e-12:
                break
        
        # Newton refinement from the midpoint
        x = (a + b) / 2.0
        for i in range(20):
            if _check_time():
                break
            fx = _f(x)
            if abs(fx) <= 1e-10:
                return x
            if abs(fx) < best_fabs:
                best_x = x
                best_fabs = abs(fx)
            
            h = 1e-8 * max(1.0, abs(x))
            d = (_f(x + h) - fx) / h
            if d == 0:
                break
            x_new = x - fx / d
            # Damping: if Newton step is too large, use bisection
            if x_new < a - (b - a) or x_new > b + (b - a):
                x_new = (a + b) / 2.0
            x = x_new
        return x
    else:
        # No bracket found; try Newton from x0
        x = x0
        for i in range(30):
            if _check_time():
                break
            fx = _f(x)
            if abs(fx) <= 1e-8:
                return x
            if abs(fx) < best_fabs:
                best_x = x
                best_fabs = abs(fx)
            
            h = 1e-7 * max(1.0, abs(x))
            d = (_f(x + h) - fx) / h
            if d == 0:
                break
            x_new = x - fx / d
            # Limit step size
            max_step = 1.0 * max(1.0, abs(x))
            if abs(x_new - x) > max_step:
                x_new = x + max_step * (1 if x_new > x else -1)
            x = x_new
        return best_x
# EVOLVE-BLOCK-END
