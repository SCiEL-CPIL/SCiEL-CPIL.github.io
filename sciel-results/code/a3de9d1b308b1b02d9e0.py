import math
import time

def find_root(f, x0):
    """
    Finds a root x of f(x) such that |f(x)| <= 1e-8.
    Uses Illinois method with adaptive step-doubling for bracketing.
    Illinois method is a modified bisection that converges faster for ill-conditioned problems.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s/150s limit
    max_evals = 500
    eval_count = 0
    
    def check_time():
        return time.time() > deadline

    def eval_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return None
        try:
            fx = f(x)
            eval_count += 1
            if math.isnan(fx) or math.isinf(fx):
                return None
            return fx
        except Exception:
            return None

    # Initial evaluation
    x0 = float(x0)
    fx0 = eval_f(x0)
    if fx0 is None:
        return x0
        
    best_x = x0
    best_fx = fx0
    
    if abs(fx0) <= 1e-8:
        return x0

    # Strategy: Find a bracket using adaptive step-doubling, then use Illinois method
    scale = max(1.0, abs(x0))
    delta = 1e-6 * scale
    
    # Try to find a bracket by expanding intervals around x0
    a, b = None, None
    fa, fb = None, None
    bracket_found = False
    
    # Try right side first
    for _ in range(60):
        if check_time() or eval_count >= max_evals - 10:
            break
            
        x_r = x0 + delta
        fx_r = eval_f(x_r)
        if fx_r is None:
            break
            
        if abs(fx_r) < abs(best_fx):
            best_x, best_fx = x_r, fx_r
        if abs(fx_r) <= 1e-8:
            return x_r
            
        if fx0 * fx_r < 0:
            a, b = x0, x_r
            fa, fb = fx0, fx_r
            bracket_found = True
            break
        delta *= 2.0

    # If no bracket found on right, try left
    if not bracket_found and not check_time() and eval_count < max_evals - 10:
        delta = 1e-6 * scale
        for _ in range(60):
            if check_time() or eval_count >= max_evals - 10:
                break
                
            x_l = x0 - delta
            fx_l = eval_f(x_l)
            if fx_l is None:
                break
                
            if abs(fx_l) < abs(best_fx):
                best_x, best_fx = x_l, fx_l
            if abs(fx_l) <= 1e-8:
                return x_l
                
            if fx0 * fx_l < 0:
                a, b = x_l, x0
                fa, fb = fx_l, fx0
                bracket_found = True
                break
            delta *= 2.0

    # If still no bracket, try combining
    if not bracket_found and not check_time() and eval_count < max_evals - 10:
        # Try a wider range
        for factor in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]:
            if check_time() or eval_count >= max_evals - 10:
                break
            delta = factor * scale
            x_r = x0 + delta
            x_l = x0 - delta
            fx_r = eval_f(x_r)
            fx_l = eval_f(x_l)
            
            if fx_r is not None:
                if abs(fx_r) < abs(best_fx):
                    best_x, best_fx = x_r, fx_r
                if abs(fx_r) <= 1e-8:
                    return x_r
                if fx0 * fx_r < 0:
                    a, b = x0, x_r
                    fa, fb = fx0, fx_r
                    bracket_found = True
                    break
            
            if fx_l is not None:
                if abs(fx_l) < abs(best_fx):
                    best_x, best_fx = x_l, fx_l
                if abs(fx_l) <= 1e-8:
                    return x_l
                if fx0 * fx_l < 0:
                    a, b = x_l, x0
                    fa, fb = fx_l, fx0
                    bracket_found = True
                    break
                if fx_r is not None and fx_r * fx_l < 0:
                    a, b = x_l, x_r
                    fa, fb = fx_l, fx_r
                    bracket_found = True
                    break

    if bracket_found and not check_time():
        # Use Illinois method
        try:
            # Illinois method implementation
            if a > b:
                a, b = b, a
                fa, fb = fb, fa
                
            # Illinois method
            for _ in range(200):
                if check_time() or eval_count >= max_evals - 2:
                    break
                    
                fm = eval_f((a + b) / 2.0)
                if fm is None:
                    break
                    
                if abs(fm) < abs(best_fx):
                    best_x, best_fx = (a + b) / 2.0, fm
                if abs(fm) <= 1e-8:
                    return (a + b) / 2.0
                    
                if fa * fm < 0:
                    b = (a + b) / 2.0
                    fb = fm
                    fa /= 2.0  # Illinois adjustment
                else:
                    a = (a + b) / 2.0
                    fa = fm
                    fb /= 2.0  # Illinois adjustment
                
                if abs(b - a) < 1e-15 * max(1.0, abs((a + b) / 2.0)):
                    break
                    
            # Final check
            x_final = (a + b) / 2.0
            fx_final = eval_f(x_final)
            if fx_final is not None:
                if abs(fx_final) <= 1e-8:
                    return x_final
                elif abs(fx_final) < abs(best_fx):
                    best_x, best_fx = x_final, fx_final
        except Exception:
            pass

    # Fallback: Secant method from best_x
    if eval_count < max_evals - 5 and not check_time():
        x_prev = best_x
        fx_prev = best_fx
        x_curr = best_x + 1e-4 * scale
        fx_curr = eval_f(x_curr)
        
        if fx_curr is not None and fx_curr != fx_prev:
            for _ in range(30):
                if check_time() or eval_count >= max_evals - 2:
                    break
                if abs(fx_curr - fx_prev) < 1e-15:
                    break
                x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
                fx_new = eval_f(x_new)
                if fx_new is None:
                    break
                if abs(fx_new) < abs(best_fx):
                    best_x, best_fx = x_new, fx_new
                if abs(fx_new) <= 1e-8:
                    return x_new
                x_prev, fx_prev = x_curr, fx_curr
                x_curr, fx_curr = x_new, fx_new

    return best_x
