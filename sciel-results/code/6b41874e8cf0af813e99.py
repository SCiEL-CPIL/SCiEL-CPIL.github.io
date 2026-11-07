"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f near x0 using a robust hybrid method:
    1. Try to bracket the root using adaptive steps.
    2. Use Illinois/bisection hybrid for guaranteed convergence.
    3. Polish with Newton/secant if possible.
    
    Budget: < 500 calls. Deadline guard for 150s wall-clock.
    """
    start_time = time.time()
    DEADLINE = 150.0 - 5.0  # safety margin
    MAX_CALLS = 490
    TOL = 1e-8
    TOL_POLISH = 1e-12
    
    call_count = [0]
    
    def safe_f(x):
        """Wrapper to count calls and handle non-finite values."""
        call_count[0] += 1
        try:
            val = f(x)
        except Exception:
            return float('nan')
        if not math.isfinite(val):
            return float('nan')
        return val
    
    def time_check():
        return time.time() - start_time < DEADLINE and call_count[0] < MAX_CALLS
    
    # Initial evaluation
    x = float(x0)
    fx = safe_f(x)
    
    if abs(fx) <= TOL:
        return x
    if not math.isfinite(fx):
        # Try a small step
        for step in [1e-6, -1e-6, 1e-3, -1e-3, 1.0, -1.0]:
            if not time_check():
                break
            x_try = x + step
            fx_try = safe_f(x_try)
            if abs(fx_try) <= TOL:
                return x_try
            if math.isfinite(fx_try):
                x = x_try
                fx = fx_try
                break
        if not math.isfinite(fx):
            return x  # Return best guess
    
    # Phase 1: Bracket the root
    # Try to find a bracket [a, b] where f(a) and f(b) have opposite signs
    a = x
    fa = fx
    b = None
    fb = None
    
    # Adaptive step sizes to find a bracket
    step_sizes = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    directions = [1, -1]
    
    for direction in directions:
        if not time_check():
            break
        for ss in step_sizes:
            if not time_check():
                break
            b_try = a + direction * ss * max(1.0, abs(a))
            fb_try = safe_f(b_try)
            if not math.isfinite(fb_try):
                continue
            if fa * fb_try < 0:
                b = b_try
                fb = fb_try
                break
            # If we found a point with smaller |f|, update a
            if abs(fb_try) < abs(fa):
                a = b_try
                fa = fb_try
        if b is not None:
            break
    
    # If no bracket found, use secant/Newton from the best point
    if b is None:
        # Use secant method with adaptive steps
        best_x = x
        best_fx = fx
        
        # Try secant with a nearby point
        h = 1e-6 * max(1.0, abs(x))
        x1 = x + h
        fx1 = safe_f(x1)
        
        if not math.isfinite(fx1):
            h = 1e-4 * max(1.0, abs(x))
            x1 = x + h
            fx1 = safe_f(x1)
        
        if math.isfinite(fx1):
            # Secant iteration
            x0_sec = x
            x1_sec = x1
            fx0 = fx
            fx1_val = fx1
            
            for _ in range(50):
                if not time_check():
                    break
                if abs(fx1_val) <= TOL:
                    return x1_sec
                if abs(fx0) <= TOL:
                    return x0_sec
                
                denom = fx1_val - fx0
                if abs(denom) < 1e-30:
                    break
                x_new = x1_sec - fx1_val * (x1_sec - x0_sec) / denom
                if not math.isfinite(x_new):
                    break
                fx_new = safe_f(x_new)
                if not math.isfinite(fx_new):
                    break
                
                x0_sec, fx0 = x1_sec, fx1_val
                x1_sec, fx1_val = x_new, fx_new
                
                if abs(fx_new) < abs(best_fx):
                    best_x = x_new
                    best_fx = fx_new
            
            if abs(best_fx) <= TOL:
                return best_x
            return best_x
        else:
            return best_x
    
    # Phase 2: Illinois method (modified bisection) for guaranteed convergence
    # a and b bracket the root: f(a) * f(b) < 0
    for _ in range(80):
        if not time_check():
            break
        
        if abs(fa) <= TOL:
            return a
        if abs(fb) <= TOL:
            return b
        
        # Illinois midpoint: weighted toward the side with smaller |f|
        if abs(fa) < abs(fb):
            c = (a * fb - b * fa) / (fb - fa)  # Linear interpolation
            # But Illinois uses: c = (a*|fb| + b*|fa|) / (|fa| + |fb|)
            c = (a * abs(fb) + b * abs(fa)) / (abs(fa) + abs(fb))
            fc = safe_f(c)
            if not math.isfinite(fc):
                c = 0.5 * (a + b)
                fc = safe_f(c)
                if not math.isfinite(fc):
                    break
            if fc == 0.0 or abs(fc) <= TOL:
                return c
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc * 0.5  # Illinois damping
        else:
            c = (a * abs(fb) + b * abs(fa)) / (abs(fa) + abs(fb))
            fc = safe_f(c)
            if not math.isfinite(fc):
                c = 0.5 * (a + b)
                fc = safe_f(c)
                if not math.isfinite(fc):
                    break
            if fc == 0.0 or abs(fc) <= TOL:
                return c
            if fb * fc < 0:
                a = c
                fa = fc
            else:
                b = c
                fb = fc * 0.5  # Illinois damping
    
    # Phase 3: Polish with Newton if we're close
    x_best = 0.5 * (a + b)
    fx_best = safe_f(x_best)
    
    if math.isfinite(fx_best) and abs(fx_best) < 1e-4:
        # Newton polish
        for _ in range(20):
            if not time_check():
                break
            if abs(fx_best) <= TOL_POLISH:
                return x_best
            
            h = 1e-8 * max(1.0, abs(x_best))
            fx_plus = safe_f(x_best + h)
            if not math.isfinite(fx_plus):
                break
            deriv = (fx_plus - fx_best) / h
            if abs(deriv) < 1e-30:
                break
            x_new = x_best - fx_best / deriv
            if not math.isfinite(x_new):
                break
            fx_new = safe_f(x_new)
            if not math.isfinite(fx_new):
                break
            if abs(fx_new) >= abs(fx_best):
                break
            x_best = x_new
            fx_best = fx_new
    
    if abs(fx_best) <= TOL:
        return x_best
    
    # Return the best point found
    if abs(fa) <= abs(fb):
        return a
    else:
        return b

# EVOLVE-BLOCK-END
