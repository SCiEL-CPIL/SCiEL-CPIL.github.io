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
    
    # Phase 1: Try Newton's method with adaptive damping
    for i in range(60):
        if not check_time() or eval_count >= MAX_EVALS:
            break
        if abs(fx) <= 1e-8:
            return x
        
        # Adaptive finite difference derivative
        base_h = 1e-7 * max(1.0, abs(x))
        h = base_h
        
        # Try multiple h values for better derivative estimate
        best_deriv = None
        best_h = None
        for h_try in [h, h * 0.1, h * 10]:
            try:
                fx_plus = call_f(x + h_try)
                d_try = (fx_plus - fx) / h_try
                if math.isfinite(d_try) and abs(d_try) > 1e-15:
                    if best_deriv is None or abs(d_try) > abs(best_deriv):
                        best_deriv = d_try
                        best_h = h_try
            except:
                continue
        
        if best_deriv is None:
            # Fallback: try central difference
            try:
                fx_plus = call_f(x + h)
                fx_minus = call_f(x - h)
                d = (fx_plus - fx_minus) / (2 * h)
                if math.isfinite(d) and abs(d) > 1e-15:
                    best_deriv = d
            except:
                break
        
        if best_deriv is None:
            break
        
        d = best_deriv
        
        # Damped Newton step with adaptive damping
        step = fx / d
        # Limit step size to avoid divergence
        max_step = max(1.0, abs(x)) * 0.5
        if abs(step) > max_step:
            step = max_step * (1.0 if step > 0 else -1.0)
        
        # Try full step first, then damped steps
        accepted = False
        for damp in [1.0, 0.5, 0.25, 0.125, 0.0625]:
            x_new = x - step * damp
            try:
                fx_new = call_f(x_new)
            except:
                break
            
            if not math.isfinite(fx_new):
                continue
            
            update_best(x_new, fx_new)
            
            # Accept if it reduces |f| significantly or if we're making progress
            if abs(fx_new) < abs(fx) * 0.9 or (i < 3 and abs(fx_new) < abs(fx)):
                x = x_new
                fx = fx_new
                accepted = True
                break
            elif abs(fx_new) < abs(fx):
                # Accept smaller improvement
                x = x_new
                fx = fx_new
                accepted = True
                break
        
        if not accepted:
            # If no step worked, try secant-like approach
            break
    
    if abs(fx) <= 1e-8:
        return x
    if abs(best_fx) <= 1e-8:
        return best_x
    
    # Phase 2: Secant method with refinement
    if eval_count < MAX_EVALS and check_time():
        x0_sec = float(x0)
        x1_sec = x
        fx0 = best_fx
        fx1 = fx
        
        # If we have a better point, use it
        if abs(best_fx) < abs(fx1):
            x1_sec = best_x
            fx1 = best_fx
        
        for i in range(40):
            if not check_time() or eval_count >= MAX_EVALS:
                break
            if abs(fx1) <= 1e-8:
                return x1_sec
            
            if abs(fx0 - fx1) < 1e-15:
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
        
        # Expand search range with multiple strategies
        step_size = max(1.0, abs(best_x))
        found_bracket = False
        b = None
        fb = None
        
        # Try expanding in both directions with various multipliers
        for direction in [1, -1]:
            for mult in [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]:
                if not check_time() or eval_count >= MAX_EVALS:
                    break
                b_candidate = a + direction * step_size * mult
                try:
                    fb_candidate = call_f(b_candidate)
                except:
                    continue
                if not math.isfinite(fb_candidate):
                    continue
                update_best(b_candidate, fb_candidate)
                if fa * fb_candidate < 0:
                    found_bracket = True
                    b = b_candidate
                    fb = fb_candidate
                    break
            if found_bracket:
                break
        
        # Also try using x from Newton as bracket endpoint
        if not found_bracket and abs(fx) < abs(fa) and eval_count < MAX_EVALS:
            a = x
            fa = fx
            for direction in [1, -1]:
                for mult in [1, 2, 4, 8, 16]:
                    if not check_time() or eval_count >= MAX_EVALS:
                        break
                    b_candidate = a + direction * step_size * mult
                    try:
                        fb_candidate = call_f(b_candidate)
                    except:
                        continue
                    if not math.isfinite(fb_candidate):
                        continue
                    update_best(b_candidate, fb_candidate)
                    if fa * fb_candidate < 0:
                        found_bracket = True
                        b = b_candidate
                        fb = fb_candidate
                        break
                if found_bracket:
                    break
        
        if found_bracket and eval_count < MAX_EVALS and check_time():
            # Bisection with early termination
            for i in range(80):
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
                # Early stop if interval is small enough
                if abs(b - a) < 1e-12 * max(1.0, abs(mid)):
                    break
            if abs(fb) <= 1e-8:
                return b
            if abs(fa) <= 1e-8:
                return a
    
    # Phase 4: Final high-precision polish if we're close
    if abs(best_fx) < 1e-4 and eval_count < MAX_EVALS and check_time():
        x = best_x
        fx = best_fx
        
        # Use secant method with tight tolerance
        x0_polish = x
        # Perturb slightly to get second point
        h_polish = 1e-6 * max(1.0, abs(x))
        x1_polish = x + h_polish
        try:
            fx1_polish = call_f(x1_polish)
        except:
            fx1_polish = None
        
        if fx1_polish is not None and math.isfinite(fx1_polish):
            for i in range(30):
                if not check_time() or eval_count >= MAX_EVALS:
                    break
                if abs(fx1_polish) <= 1e-8:
                    return x1_polish
                
                if abs(fx - fx1_polish) < 1e-15:
                    break
                
                x_new = x1_polish - fx1_polish * (x1_polish - x0_polish) / (fx1_polish - fx)
                try:
                    fx_new = call_f(x_new)
                except:
                    break
                
                if not math.isfinite(fx_new):
                    break
                
                update_best(x_new, fx_new)
                
                x0_polish, fx = x1_polish, fx1_polish
                x1_polish, fx1_polish = x_new, fx_new
            
            if abs(fx1_polish) <= 1e-8:
                return x1_polish
    
    # Return best point found
    return best_x
# EVOLVE-BLOCK-END
