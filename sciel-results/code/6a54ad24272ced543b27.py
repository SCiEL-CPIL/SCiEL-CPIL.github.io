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
    DEADLINE = 130.0  # safety margin before 150s kill
    
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
    except Exception:
        return best_x
    
    def update_best(x, fx):
        nonlocal best_x, best_fx
        if fx is not None and (best_fx is None or abs(fx) < abs(best_fx)):
            best_x = x
            best_fx = fx
    
    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x
    
    x = float(x0)
    
    # Phase 1: Try to find a bracket using expansion
    a, b, fa, fb = None, None, None, None
    if best_fx is not None:
        a, fa = x, best_fx
        # Try expanding to find a sign change - use adaptive step
        step = max(1e-3, abs(x) * 1e-3)
        for i in range(30):
            if not check_time() or eval_count >= MAX_EVALS:
                break
            # Alternate direction to increase chance of finding a bracket
            direction = 1 if (i % 2 == 0) else -1
            b = a + direction * step * (2 ** i)
            try:
                fb = call_f(b)
                update_best(b, fb)
                if abs(fb) <= 1e-8:
                    return b
                if fa * fb < 0:
                    break
                a, fa = b, fb
            except Exception:
                break
        
        if fa is not None and fb is not None and fa * fb < 0:
            # We have a bracket, use bisection with acceleration
            for _ in range(150):
                if not check_time() or eval_count >= MAX_EVALS:
                    break
                mid = (a + b) / 2.0
                try:
                    fm = call_f(mid)
                    update_best(mid, fm)
                    if abs(fm) <= 1e-8:
                        return mid
                    if fa * fm < 0:
                        b, fb = mid, fm
                    else:
                        a, fa = mid, fm
                    if abs(b - a) < 1e-12 * max(1.0, abs(a)):
                        break
                except Exception:
                    break
            # Return the better of a, b
            if abs(fa) <= abs(fb):
                return a
            else:
                return b
    
    # Phase 2: Secant-based iteration (avoids extra evals for derivative)
    x_prev = x
    x_curr = x + 1e-5 * max(1.0, abs(x))
    
    try:
        fx_prev = call_f(x_prev)
        fx_curr = call_f(x_curr)
        update_best(x_prev, fx_prev)
        update_best(x_curr, fx_curr)
    except Exception:
        fx_prev = best_fx
        fx_curr = None
    
    if fx_curr is None:
        # Try different perturbation
        x_curr = x - 1e-5 * max(1.0, abs(x))
        try:
            fx_curr = call_f(x_curr)
            update_best(x_curr, fx_curr)
        except Exception:
            fx_curr = None
    
    if fx_curr is None:
        return best_x
    
    # Main iteration loop: secant with fallback to damped Newton
    for iteration in range(250):
        if not check_time() or eval_count >= MAX_EVALS:
            break
        if best_fx is not None and abs(best_fx) <= 1e-8:
            return best_x
        
        chosen_x = None
        
        # Secant step (primary method - no extra evals)
        if fx_prev is not None and fx_curr is not None and abs(fx_curr - fx_prev) > 1e-30:
            secant_x = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
            if math.isfinite(secant_x):
                max_step = 5.0 * max(1.0, abs(x_curr))
                if abs(secant_x - x_curr) <= max_step:
                    chosen_x = secant_x
                else:
                    direction = 1 if secant_x > x_curr else -1
                    chosen_x = x_curr + direction * max_step
        
        # If secant failed, try Newton with finite difference (costs 1 eval)
        if chosen_x is None:
            h = 1e-6 * max(1.0, abs(x_curr))
            try:
                fx_plus = call_f(x_curr + h)
                update_best(x_curr + h, fx_plus)
                if abs(fx_plus - fx_curr) > 1e-30:
                    dfdx = (fx_plus - fx_curr) / h
                    if abs(dfdx) > 1e-30:
                        newton_x = x_curr - fx_curr / dfdx
                        if math.isfinite(newton_x):
                            max_step = 5.0 * max(1.0, abs(x_curr))
                            if abs(newton_x - x_curr) <= max_step:
                                chosen_x = newton_x
                            else:
                                direction = 1 if newton_x > x_curr else -1
                                chosen_x = x_curr + direction * max_step
            except Exception:
                pass
        
        if chosen_x is None:
            # Fallback: small perturbation
            chosen_x = x_curr + 1e-3 * max(1.0, abs(x_curr))
        
        if not math.isfinite(chosen_x):
            break
        
        try:
            fx_new = call_f(chosen_x)
            update_best(chosen_x, fx_new)
        except Exception:
            break
        
        if abs(fx_new) <= 1e-8:
            return chosen_x
        
        # Update for next iteration
        x_prev = x_curr
        fx_prev = fx_curr
        x_curr = chosen_x
        fx_curr = fx_new
        
        # Check convergence
        if abs(x_curr - x_prev) < 1e-12 * max(1.0, abs(x_curr)):
            break
    
    # Phase 3: Final polish - if we're close, try to tighten
    if best_fx is not None and abs(best_fx) < 1e-4:
        # Use more precise local search with secant
        x_local = best_x
        fx_local = best_fx
        x_local_prev = x_local - 1e-7 * max(1.0, abs(x_local))
        try:
            fx_local_prev = call_f(x_local_prev)
            update_best(x_local_prev, fx_local_prev)
        except Exception:
            fx_local_prev = None
        
        for _ in range(80):
            if not check_time() or eval_count >= MAX_EVALS:
                break
            if abs(fx_local) <= 1e-8:
                return x_local
            
            # Try secant step if we have two points
            if fx_local_prev is not None and abs(fx_local - fx_local_prev) > 1e-30:
                secant_x = x_local - fx_local * (x_local - x_local_prev) / (fx_local - fx_local_prev)
                if math.isfinite(secant_x):
                    try:
                        fx_try = call_f(secant_x)
                        update_best(secant_x, fx_try)
                        if abs(fx_try) <= 1e-8:
                            return secant_x
                        if abs(fx_try) < abs(fx_local) * 0.5:
                            x_local_prev = x_local
                            fx_local_prev = fx_local
                            x_local = secant_x
                            fx_local = fx_try
                            continue
                    except Exception:
                        pass
            
            # Try small steps in both directions
            h = 1e-8 * max(1.0, abs(x_local))
            improved = False
            for direction in [1, -1]:
                x_try = x_local + direction * h
                try:
                    fx_try = call_f(x_try)
                    update_best(x_try, fx_try)
                    if abs(fx_try) <= 1e-8:
                        return x_try
                    if abs(fx_try) < abs(fx_local) * 0.5:
                        x_local = x_try
                        fx_local = fx_try
                        improved = True
                except Exception:
                    pass
            
            if not improved:
                h *= 0.5
            if h < 1e-14:
                break
    
    return best_x
# EVOLVE-BLOCK-END
