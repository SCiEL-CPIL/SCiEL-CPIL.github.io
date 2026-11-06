"""Root finding for ill-conditioned problems."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find root of f near x0 using a robust hybrid approach.
    Returns x with |f(x)| <= 1e-8 or best effort within 500 evals.
    """
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    eval_count = 0
    MAX_EVALS = 500
    
    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        result = f(x)
        eval_count += 1
        return result
    
    def check_time():
        return (time.time() - start_time) < DEADLINE
    
    # Track best point found
    best_x = float(x0)
    best_fx = None
    try:
        best_fx = call_f(best_x)
    except:
        best_fx = float('inf')
    
    def update_best(x, fx):
        nonlocal best_x, best_fx
        if abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
    
    update_best(best_x, best_fx)
    
    if abs(best_fx) <= 1e-8:
        return best_x
    
    # Try Newton with damped steps, bisection fallback, and secant
    x = float(x0)
    fx = best_fx
    
    # Phase 1: Try Newton's method with damping
    for i in range(50):
        if not check_time() or eval_count >= MAX_EVALS:
            break
        if abs(fx) <= 1e-8:
            return x
        
        # Finite difference derivative
        h = 1e-8 * max(1.0, abs(x))
        try:
            fx_plus = call_f(x + h)
        except:
            break
        
        d = (fx_plus - fx) / h
        
        if d == 0 or not math.isfinite(d):
            break
        
        # Damped Newton step
        step = fx / d
        # Limit step size to avoid divergence
        max_step = max(1.0, abs(x)) * 0.5
        if abs(step) > max_step:
            step = max_step * (1.0 if step > 0 else -1.0)
        
        x_new = x - step
        try:
            fx_new = call_f(x_new)
        except:
            break
        
        if not math.isfinite(fx_new):
            break
        
        update_best(x_new, fx_new)
        
        # Accept if it reduces |f| or if we're making progress
        if abs(fx_new) < abs(fx) or i < 5:
            x = x_new
            fx = fx_new
        else:
            # Try smaller step
            step = step * 0.5
            x_new = x - step
            try:
                fx_new = call_f(x_new)
            except:
                break
            if math.isfinite(fx_new) and abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
                update_best(x, fx)
    
    if abs(fx) <= 1e-8:
        return x
    if abs(best_fx) <= 1e-8:
        return best_x
    
    # Phase 2: Secant method
    if eval_count < MAX_EVALS and check_time():
        x0_sec = float(x0)
        x1_sec = x
        fx0 = best_fx
        fx1 = fx
        
        for i in range(30):
            if not check_time() or eval_count >= MAX_EVALS:
                break
            if abs(fx1) <= 1e-8:
                return x1_sec
            
            if fx0 == fx1:
                break
            
            x_new = x1_sec - fx1 * (x1_sec - x0_sec) / (fx1 - fx0)
            try:
                fx_new = call_f(x_new)
            except:
                break
            
            if not math.isfinite(fx_new):
                break
            
            update_best(x_new, fx_new)
            
            x0_sec, fx0 = x1_sec, fx1
            x1_sec, fx1 = x_new, fx_new
        
        if abs(fx1) <= 1e-8:
            return x1_sec
        if abs(best_fx) <= 1e-8:
            return best_x
    
    # Phase 3: Bisection if we can bracket the root
    if eval_count < MAX_EVALS and check_time():
        # Try to find a bracket around the best point
        a = best_x
        fa = best_fx
        
        # Expand search range
        step_size = max(1.0, abs(best_x))
        found_bracket = False
        for direction in [1, -1]:
            for mult in [1, 2, 4, 8, 16, 32, 64, 128]:
                if not check_time() or eval_count >= MAX_EVALS:
                    break
                b = a + direction * step_size * mult
                try:
                    fb = call_f(b)
                except:
                    continue
                if not math.isfinite(fb):
                    continue
                update_best(b, fb)
                if fa * fb < 0:
                    found_bracket = True
                    break
            if found_bracket:
                break
        
        if found_bracket and eval_count < MAX_EVALS and check_time():
            # Bisection
            for i in range(60):
                if not check_time() or eval_count >= MAX_EVALS:
                    break
                mid = (a + b) / 2.0
                try:
                    fmid = call_f(mid)
                except:
                    break
                if not math.isfinite(fmid):
                    break
                update_best(mid, fmid)
                if abs(fmid) <= 1e-8:
                    return mid
                if fa * fmid < 0:
                    b = mid
                    fb = fmid
                else:
                    a = mid
                    fa = fmid
            if abs(fb) <= 1e-8:
                return b
            if abs(fa) <= 1e-8:
                return a
    
    # Return best point found
    return best_x
# EVOLVE-BLOCK-END
