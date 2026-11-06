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
    DEADLINE = 135.0  # safety margin before 150s kill
    
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
        if fx is not None and abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
        if best_fx is not None and abs(best_fx) <= 1e-8:
            return True
        return False
    
    update_best(best_x, best_fx)
    
    x = float(x0)
    fx = best_fx
    
    # Phase 1: Try to establish a bracket around x0
    # Use expanding steps to find sign change
    a, b = None, None
    fa, fb = None, None
    
    # Try small steps first
    step = 1e-4 * max(1.0, abs(x0))
    for i in range(20):
        if not check_time() or eval_count >= MAX_EVALS:
            break
        x_try = x0 + step
        try:
            fx_try = call_f(x_try)
        except Exception:
            step *= -2
            continue
        update_best(x_try, fx_try)
        if fx_try is None or math.isnan(fx_try) or math.isinf(fx_try):
            step *= -2
            continue
        if fx * fx_try < 0:
            a, b = x0, x_try
            fa, fb = fx, fx_try
            break
        # Also check the other direction
        x_try2 = x0 - step
        try:
            fx_try2 = call_f(x_try2)
        except Exception:
            step *= 2
            continue
        update_best(x_try2, fx_try2)
        if fx_try2 is None or math.isnan(fx_try2) or math.isinf(fx_try2):
            step *= 2
            continue
        if fx * fx_try2 < 0:
            a, b = x_try2, x0
            fa, fb = fx_try2, fx
            break
        step *= 2
    
    # Phase 2: If we have a bracket, use bisection + secant hybrid
    if a is not None and b is not None:
        # Use Illinois-like method or bisection with secant acceleration
        for _ in range(100):
            if not check_time() or eval_count >= MAX_EVALS:
                break
            # Bisection midpoint
            m = (a + b) / 2.0
            try:
                fm = call_f(m)
            except Exception:
                break
            update_best(m, fm)
            if fm is None or math.isnan(fm) or math.isinf(fm):
                break
            if abs(fm) <= 1e-8:
                return m
            if fa * fm < 0:
                b = m
                fb = fm
            else:
                a = m
                fa = fm
            # Check if we can use secant step
            if fa * fb < 0 and fa != fb:
                # Secant step
                x_sec = a - fa * (b - a) / (fb - fa)
                if a < x_sec < b:
                    try:
                        f_sec = call_f(x_sec)
                    except Exception:
                        continue
                    update_best(x_sec, f_sec)
                    if f_sec is None or math.isnan(f_sec) or math.isinf(f_sec):
                        continue
                    if abs(f_sec) <= 1e-8:
                        return x_sec
                    if fa * f_sec < 0:
                        b = x_sec
                        fb = f_sec
                    else:
                        a = x_sec
                        fa = f_sec
    
    # Phase 3: Newton's method with damping
    x = best_x
    try:
        fx = call_f(x)
    except Exception:
        fx = best_fx
    update_best(x, fx)
    
    for _ in range(80):
        if not check_time() or eval_count >= MAX_EVALS:
            break
        if fx is None or math.isnan(fx) or math.isinf(fx):
            break
        if abs(fx) <= 1e-8:
            return x
        # Finite difference derivative
        h = 1e-7 * max(1.0, abs(x))
        try:
            fx_h = call_f(x + h)
        except Exception:
            break
        if fx_h is None or math.isnan(fx_h) or math.isinf(fx_h):
            break
        d = (fx_h - fx) / h
        if d == 0 or math.isnan(d) or math.isinf(d):
            break
        # Damped Newton step
        dx = -fx / d
        # Limit step size
        max_step = 10.0 * max(1.0, abs(x))
        if abs(dx) > max_step:
            dx = max_step if dx > 0 else -max_step
        x_new = x + dx
        try:
            fx_new = call_f(x_new)
        except Exception:
            break
        update_best(x_new, fx_new)
        if fx_new is None or math.isnan(fx_new) or math.isinf(fx_new):
            # Halve the step
            dx *= 0.5
            x_new = x + dx
            try:
                fx_new = call_f(x_new)
            except Exception:
                break
            update_best(x_new, fx_new)
            if fx_new is None or math.isnan(fx_new) or math.isinf(fx_new):
                break
        if abs(fx_new) < abs(fx):
            x = x_new
            fx = fx_new
        else:
            # Accept anyway if it's not worse by too much, or break
            if abs(fx_new) < 10 * abs(fx):
                x = x_new
                fx = fx_new
            else:
                break
    
    # Phase 4: Secant method as fallback
    if eval_count < MAX_EVALS and check_time():
        x0_sec = best_x
        try:
            fx0 = call_f(x0_sec)
        except Exception:
            fx0 = best_fx
        h = 1e-5 * max(1.0, abs(x0_sec))
        x1_sec = x0_sec + h
        try:
            fx1 = call_f(x1_sec)
        except Exception:
            fx1 = None
        if fx1 is not None and not math.isnan(fx1) and not math.isinf(fx1):
            for _ in range(50):
                if not check_time() or eval_count >= MAX_EVALS:
                    break
                if fx0 is None or fx1 is None:
                    break
                if abs(fx1) <= 1e-8:
                    return x1_sec
                if abs(fx0) <= 1e-8:
                    return x0_sec
                if fx0 == fx1:
                    break
                x2_sec = x1_sec - fx1 * (x1_sec - x0_sec) / (fx1 - fx0)
                try:
                    fx2 = call_f(x2_sec)
                except Exception:
                    break
                update_best(x2_sec, fx2)
                if fx2 is None or math.isnan(fx2) or math.isinf(fx2):
                    break
                if abs(fx2) <= 1e-8:
                    return x2_sec
                x0_sec, fx0 = x1_sec, fx1
                x1_sec, fx1 = x2_sec, fx2
    
    return best_x
# EVOLVE-BLOCK-END
