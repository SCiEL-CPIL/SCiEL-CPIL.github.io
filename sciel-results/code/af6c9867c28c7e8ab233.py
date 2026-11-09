import time
import math

def find_root(f, x0):
    """
    Finds a root x of f such that |f(x)| <= 1e-8.
    Uses a hybrid Newton-Bisection method with adaptive step damping.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    eval_count = 0
    max_evals = 500
    
    def safe_f(x):
        nonlocal eval_count
        if time.time() > deadline:
            raise TimeoutError("Time limit exceeded")
        if eval_count >= max_evals:
            raise ValueError("Max evaluations exceeded")
        eval_count += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return val
            return float(val)
        except Exception:
            return float('nan')

    x = float(x0)
    fx = safe_f(x)
    
    if abs(fx) <= 1e-8:
        return x

    # Try to find a bracket [a, b]
    a = x
    b = x
    fa = fx
    fb = fx
    
    # Expand to find sign change
    # We will try to expand b and a
    bracket_found = False
    # Start with small expansion
    step = 1.0
    for _ in range(20):
        if time.time() > deadline:
            break
        # Try expanding b
        b_new = x + step
        fb_new = safe_f(b_new)
        if math.isfinite(fb_new) and fa * fb_new < 0:
            a, b = x, b_new
            fa, fb = fx, fb_new
            bracket_found = True
            break
        
        # Try expanding a
        a_new = x - step
        fa_new = safe_f(a_new)
        if math.isfinite(fa_new) and fa_new * fx < 0:
            a, b = a_new, x
            fa, fb = fa_new, fx
            bracket_found = True
            break
            
        step *= 2.0

    # Main loop
    best_x = x
    best_fx = fx
    
    for _ in range(max_evals):
        if time.time() > deadline:
            break
        if abs(fx) <= 1e-8:
            return x
            
        # Estimate derivative using central difference if possible, else forward
        # Adaptive h
        h = 1e-5 * max(1.0, abs(x))
        
        # If we are in a bracket, we can be more aggressive or cautious
        if bracket_found:
            # Newton step
            # Use central difference for derivative if we have budget
            # f'(x) ~ (f(x+h) - f(x-h)) / (2h)
            # But this costs 2 evals. Forward costs 1.
            # Let's use forward difference to save evals, but it's less accurate.
            # Given "ill-conditioned", accuracy of derivative is key.
            # Let's try central difference.
            
            # Check budget for 2 evals
            if eval_count + 2 <= max_evals and time.time() < deadline - 1.0:
                xh = x + h
                xh_neg = x - h
                fxh = safe_f(xh)
                fxh_neg = safe_f(xh_neg)
                if math.isfinite(fxh) and math.isfinite(fxh_neg):
                    d = (fxh - fxh_neg) / (2 * h)
                else:
                    d = 0
            else:
                xh = x + h
                fxh = safe_f(xh)
                if math.isfinite(fxh):
                    d = (fxh - fx) / h
                else:
                    d = 0
            
            if d == 0 or not math.isfinite(d):
                # Fallback to bisection
                x_new = (a + b) / 2.0
                fx_new = safe_f(x_new)
                if not math.isfinite(fx_new):
                    break
                if abs(fx_new) < abs(fx):
                    fx, x = fx_new, x_new
                    if abs(fx) <= 1e-8:
                        return x
                # Update bracket
                if fa * fx_new < 0:
                    b = x_new
                    fb = fx_new
                else:
                    a = x_new
                    fa = fx_new
                continue
            
            x_new = x - fx / d
            
            # Check if x_new is in bracket
            if a <= x_new <= b:
                fx_new = safe_f(x_new)
                if math.isfinite(fx_new) and abs(fx_new) < abs(fx):
                    # Accept Newton step
                    x, fx = x_new, fx_new
                    if abs(fx) <= 1e-8:
                        return x
                    # Update bracket
                    if fa * fx < 0:
                        b = x
                        fb = fx
                    else:
                        a = x
                        fa = fx
                else:
                    # Newton step failed or didn't reduce error, use bisection
                    x_new = (a + b) / 2.0
                    fx_new = safe_f(x_new)
                    if not math.isfinite(fx_new):
                        break
                    x, fx = x_new, fx_new
                    if abs(fx) <= 1e-8:
                        return x
                    if fa * fx < 0:
                        b = x
                        fb = fx
                    else:
                        a = x
                        fa = fx
            else:
                # Newton step outside bracket, use bisection
                x_new = (a + b) / 2.0
                fx_new = safe_f(x_new)
                if not math.isfinite(fx_new):
                    break
                x, fx = x_new, fx_new
                if abs(fx) <= 1e-8:
                    return x
                if fa * fx < 0:
                    b = x
                    fb = fx
                else:
                    a = x
                    fa = fx
        else:
            # No bracket found, use damped Newton
            # Try to find derivative
            h = 1e-5 * max(1.0, abs(x))
            if eval_count + 1 <= max_evals and time.time() < deadline - 1.0:
                xh = x + h
                fxh = safe_f(xh)
                if math.isfinite(fxh):
                    d = (fxh - fx) / h
                else:
                    d = 0
            else:
                d = 0
            
            if d == 0 or not math.isfinite(d):
                # Can't compute derivative, try random jitter? Or just break?
                # If we can't compute derivative, we are stuck.
                # Try to find a bracket around current x
                step = 1e-3 * max(1.0, abs(x))
                found = False
                for _ in range(5):
                    x_plus = x + step
                    x_minus = x - step
                    fx_plus = safe_f(x_plus)
                    fx_minus = safe_f(x_minus)
                    if math.isfinite(fx_plus) and math.isfinite(fx_minus):
                        if fx * fx_plus < 0:
                            a, b = x, x_plus
                            fa, fb = fx, fx_plus
                            bracket_found = True
                            found = True
                            break
                        elif fx * fx_minus < 0:
                            a, b = x_minus, x
                            fa, fb = fx_minus, fx
                            bracket_found = True
                            found = True
                            break
                    step *= 2.0
                if found:
                    continue
                else:
                    break
            
            # Damped Newton
            step_factor = 1.0
            for _ in range(5):
                x_new = x - step_factor * fx / d
                fx_new = safe_f(x_new)
                if math.isfinite(fx_new) and abs(fx_new) < abs(fx):
                    x, fx = x_new, fx_new
                    if abs(fx) <= 1e-8:
                        return x
                    break
                step_factor *= 0.5
            
            # If we didn't accept a step, try to find a bracket
            if abs(fx) >= abs(best_fx):
                # No improvement, try to bracket
                step = 1e-3 * max(1.0, abs(x))
                for _ in range(5):
                    x_plus = x + step
                    x_minus = x - step
                    fx_plus = safe_f(x_plus)
                    fx_minus = safe_f(x_minus)
                    if math.isfinite(fx_plus) and math.isfinite(fx_minus):
                        if fx * fx_plus < 0:
                            a, b = x, x_plus
                            fa, fb = fx, fx_plus
                            bracket_found = True
                            break
                        elif fx * fx_minus < 0:
                            a, b = x_minus, x
                            fa, fb = fx_minus, fx
                            bracket_found = True
                            break
                    step *= 2.0
                if bracket_found:
                    continue
            if abs(fx) < abs(best_fx):
                best_x = x
                best_fx = fx

    # Return best found
    if abs(fx) < abs(best_fx):
        return x
    return best_x
