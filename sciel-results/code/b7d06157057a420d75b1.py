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
        # Try expanding to find a sign change
        # Increased initial step and iterations to catch roots in ill-conditioned/flat regions
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
            # We have a bracket, use bisection with high precision polish
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
    
    # Phase 2: Robust Secant (Pegasus Method)
    # Pegasus is a damped secant method that is more robust for ill-conditioned functions
    # because it avoids large steps that can lead to divergence.
    
    # Initialize two points
    x_prev = x
    x_curr = x + 1e-4 * max(1.0, abs(x))
    
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
        x_curr = x - 1e-4 * max(1.0, abs(x))
        try:
            fx_curr = call_f(x_curr)
            update_best(x_curr, fx_curr)
        except Exception:
            fx_curr = None
    
    if fx_curr is None:
        return best_x
    
    # Main iteration loop using Pegasus
    for iteration in range(200):
        if not check_time() or eval_count >= MAX_EVALS:
            break
        if best_fx is not None and abs(best_fx) <= 1e-8:
            return best_x
        
        # Pegasus method step
        # If fx_prev and fx_curr have the same sign, the root is not bracketed by them.
        # We damp the step by replacing the point that is "farther" from the root
        # (heuristic: the one with larger |f|) with the new point, but we move
        # the new point closer to the one with smaller |f|.
        
        if fx_prev * fx_curr > 0:
            # Same sign. The root is likely beyond x_curr or x_prev.
            # We keep the point with smaller |f| as the "anchor" and move the other one.
            if abs(fx_prev) < abs(fx_curr):
                # x_prev is better. We move x_curr closer to x_prev.
                # Standard secant step would be:
                # x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
                # But we damp it.
                # Pegasus rule: replace the point with larger |f| with the new point,
                # but ensure the new point is between the old two? No, Pegasus replaces
                # the point that is on the same side as the previous step's failure.
                
                # Simple robust damping:
                # If we are moving away from the minimum of |f|, we step back.
                # Let's use the secant step but limit the distance.
                if abs(fx_curr - fx_prev) > 1e-30:
                    x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
                else:
                    x_new = x_curr + 1e-6 * max(1.0, abs(x_curr))
                
                # Damping: if x_new is further from x_prev than x_curr was, damp it.
                # Or simply: if the sign didn't change, we might be diverging.
                # Let's try a simple damped secant:
                # x_new = x_curr - 0.5 * fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
                
                # Actually, standard Pegasus:
                # If f(x_prev) and f(x_curr) have same sign, replace x_prev with x_curr
                # but move x_curr halfway between x_prev and x_curr? No.
                
                # Let's stick to a reliable damped secant:
                if abs(fx_curr - fx_prev) > 1e-30:
                    step_secant = -fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
                    # Damp the step if it's too large or if we suspect divergence
                    if abs(step_secant) > 10 * abs(x_curr - x_prev):
                        step_secant = 0.5 * step_secant
                    x_new = x_curr + step_secant
                else:
                    x_new = x_curr + 1e-6 * max(1.0, abs(x_curr))
                
                # Update: keep x_prev (better) and replace x_curr
                x_prev, fx_prev = x_curr, fx_curr
                x_curr, fx_curr = x_new, None # We need to eval x_new
                
            else:
                # x_curr is better. We move x_prev closer to x_curr.
                if abs(fx_curr - fx_prev) > 1e-30:
                    step_secant = -fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
                    if abs(step_secant) > 10 * abs(x_curr - x_prev):
                        step_secant = 0.5 * step_secant
                    x_new = x_curr + step_secant
                else:
                    x_new = x_curr + 1e-6 * max(1.0, abs(x_curr))
                
                # Update: keep x_curr (better) and replace x_prev
                x_prev, fx_prev = x_new, None
                # x_curr stays the same
        
        else:
            # Sign change! We have a bracket. Use Bisection for safety and precision.
            # This handles the case where we found a bracket during the secant process.
            a, b = x_prev, x_curr
            fa, fb = fx_prev, fx_curr
            if fa * fb > 0:
                 # This shouldn't happen if we checked sign, but just in case
                 a, b = x_curr, x_prev
                 fa, fb = fx_curr, fx_prev
            
            # Perform a few bisection steps to converge
            for _ in range(50):
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
            
            # After bisection, return best
            if abs(fa) <= abs(fb):
                return a
            else:
                return b
        
        # Evaluate the new point
        try:
            if fx_curr is None:
                fx_curr = call_f(x_curr)
                update_best(x_curr, fx_curr)
            else:
                fx_prev = call_f(x_prev)
                update_best(x_prev, fx_prev)
        except Exception:
            break
        
        if abs(fx_curr) <= 1e-8:
            return x_curr
        if abs(fx_prev) <= 1e-8:
            return x_prev
        
        # Check convergence
        if abs(x_curr - x_prev) < 1e-12 * max(1.0, abs(x_curr)):
            break
    
    # Final polish: if we are close, try Newton
    if best_fx is not None and abs(best_fx) < 1e-4:
        x_newton = best_x
        for _ in range(10):
            if not check_time() or eval_count >= MAX_EVALS:
                break
            h = 1e-7 * max(1.0, abs(x_newton))
            try:
                f_plus = call_f(x_newton + h)
                update_best(x_newton + h, f_plus)
                dfdx = (f_plus - best_fx) / h
                if abs(dfdx) > 1e-15:
                    x_new = x_newton - best_fx / dfdx
                    if math.isfinite(x_new):
                        f_new = call_f(x_new)
                        update_best(x_new, f_new)
                        if abs(f_new) <= 1e-8:
                            return x_new
                        x_newton = x_new
                        best_fx = f_new
                        best_x = x_new
                    else:
                        break
                else:
                    break
            except Exception:
                break
    
    return best_x
# EVOLVE-BLOCK-END
