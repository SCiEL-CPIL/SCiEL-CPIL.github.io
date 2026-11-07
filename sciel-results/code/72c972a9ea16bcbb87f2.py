import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 118.0
    max_calls = 490
    
    calls = 0
    best_x = float(x0)
    best_abs_f = float('inf')
    
    def _f(x):
        nonlocal calls, best_x, best_abs_f
        if calls >= max_calls:
            return float('nan')
        if time.time() - start_time > TIME_LIMIT:
            return float('nan')
        calls += 1
        try:
            val = f(x)
        except:
            return float('nan')
        if math.isfinite(val):
            abs_val = abs(val)
            if abs_val < best_abs_f:
                best_abs_f = abs_val
                best_x = x
        return val

    x = float(x0)
    fx = _f(x)
    if math.isnan(fx):
        return best_x
    if abs(fx) <= 1e-8:
        return x

    # Bracketing
    a, b, fa, fb = None, None, None, None
    found_bracket = False
    
    # Determine initial step size based on x0 magnitude
    if abs(x) > 1e6:
        step = 1e6
    elif abs(x) > 1e3:
        step = 1e3
    elif abs(x) > 1:
        step = 1.0
    else:
        step = 0.1

    # Expand in both directions
    # We try to find a sign change.
    # We keep track of the current point and its value.
    # We move outwards.
    
    # Left side
    x_left = x
    f_left = fx
    # Right side
    x_right = x
    f_right = fx
    
    # We will expand step by step.
    # To save calls, we can check x + step and x - step in each iteration?
    # No, we need to maintain the bracket.
    # Let's just do a standard expansion.
    
    # Try small steps first
    for i in range(1, 25):
        if time.time() - start_time > TIME_LIMIT - 5.0 or calls >= max_calls - 20:
            break
            
        # Try right
        x_new = x + step
        f_new = _f(x_new)
        if not math.isnan(f_new):
            if abs(f_new) <= 1e-8:
                return x_new
            if fx * f_new < 0:
                a, b, fa, fb = x, x_new, fx, f_new
                found_bracket = True
                break
            # If no sign change, update x and fx to move the window?
            # No, that's not how bracketing works usually.
            # We should expand the window.
            # If we don't find a bracket, we should try larger steps.
            # But we also need to check the other side.
            
        # Try left
        x_new_left = x - step
        f_new_left = _f(x_new_left)
        if not math.isnan(f_new_left):
            if abs(f_new_left) <= 1e-8:
                return x_new_left
            if fx * f_new_left < 0:
                a, b, fa, fb = x_new_left, x, f_new_left, fx
                found_bracket = True
                break
        
        step *= 2.0

    if found_bracket:
        try:
            # Use Brent's method
            root = brentq(_f, a, b, xtol=1e-14, rtol=1e-14, maxiter=50)
            if math.isfinite(root):
                f_root = _f(root)
                if not math.isnan(f_root) and abs(f_root) <= 1e-8:
                    return root
                # If not close enough, try a few secant steps
                if not math.isnan(f_root) and abs(f_root) < 1e-4:
                    # Secant polish
                    x1 = root
                    f1 = f_root
                    # Need another point. Use a or b? Or a small perturbation?
                    # Perturbation is safer to avoid re-using a point with potentially larger error.
                    h = 1e-5 * max(1.0, abs(root))
                    x2 = root + h
                    f2 = _f(x2)
                    if not math.isnan(f2):
                        for _ in range(10):
                            if time.time() - start_time > TIME_LIMIT - 2.0 or calls >= max_calls - 2:
                                break
                            if abs(f1 - f2) < 1e-30:
                                break
                            x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
                            if not math.isfinite(x_new):
                                break
                            f_new = _f(x_new)
                            if math.isnan(f_new):
                                break
                            if abs(f_new) <= 1e-8:
                                return x_new
                            # Update
                            x2, f2 = x1, f1
                            x1, f1 = x_new, f_new
                            if abs(x1 - x2) < 1e-14:
                                break
                        if time.time() - start_time < TIME_LIMIT - 1.0 and calls < max_calls:
                            f_final = _f(x1)
                            if not math.isnan(f_final) and abs(f_final) <= 1e-8:
                                return x1
                            if not math.isnan(f_final) and abs(f_final) < best_abs_f:
                                best_x = x1
                                best_abs_f = abs(f_final)
        except:
            pass
        
        # Fallback: Illinois method or simple bisection if brentq failed
        if time.time() - start_time < TIME_LIMIT - 5.0 and calls < max_calls - 10:
            for i in range(50):
                if time.time() - start_time > TIME_LIMIT - 2.0 or calls >= max_calls - 2:
                    break
                if abs(fa) <= 1e-8: return a
                if abs(fb) <= 1e-8: return b
                
                # Secant step
                if abs(fa - fb) > 1e-30:
                    m = (a * fb - b * fa) / (fb - fa)
                else:
                    m = (a + b) / 2
                    
                if a < m < b:
                    fm = _f(m)
                    if math.isnan(fm): break
                    if abs(fm) <= 1e-8: return m
                    if fa * fm < 0:
                        b, fb = m, fm
                    else:
                        a, fa = m, fm
                else:
                    # Bisection
                    m = (a + b) / 2
                    fm = _f(m)
                    if math.isnan(fm): break
                    if abs(fm) <= 1e-8: return m
                    if fa * fm < 0:
                        b, fb = m, fm
                    else:
                        a, fa = m, fm
            
            result_x = a if abs(fa) < abs(fb) else b
            if time.time() - start_time < TIME_LIMIT - 1.0 and calls < max_calls:
                f_final = _f(result_x)
                if not math.isnan(f_final) and abs(f_final) <= 1e-8:
                    return result_x
                if not math.isnan(f_final) and abs(f_final) < best_abs_f:
                    best_x = result_x
                    best_abs_f = abs(f_final)
        
        return best_x if best_abs_f < abs(_f(result_x)) else result_x

    # If no bracket found, do local search
    # This is a fallback for cases where the function doesn't change sign near x0
    # or we missed the bracket.
    if time.time() - start_time < TIME_LIMIT - 10.0 and calls < max_calls - 30:
        # Try a few points
        candidates = [x]
        for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]:
            for sign in [1, -1]:
                if abs(x) > 1e6:
                    step = 1e6 * scale
                elif abs(x) > 1e3:
                    step = 1e3 * scale
                elif abs(x) > 1:
                    step = scale
                else:
                    step = scale * 0.1
                candidates.append(x + sign * step)
        
        for cx in candidates:
            if time.time() - start_time > TIME_LIMIT - 5.0 or calls >= max_calls - 10:
                break
            cf = _f(cx)
            if math.isnan(cf):
                continue
            if abs(cf) <= 1e-8:
                return cx
        
        # Refine best candidate
        if best_abs_f < 1.0:
            # Secant/Newton polish
            nx = best_x
            # Get derivative estimate
            h = max(1e-8 * abs(nx), 1e-12)
            fn = _f(nx)
            if not math.isnan(fn):
                fn_plus = _f(nx + h)
                fn_minus = _f(nx - h)
                if not math.isnan(fn_plus) and not math.isnan(fn_minus):
                    df = (fn_plus - fn_minus) / (2 * h)
                    if abs(df) > 1e-30:
                        for _ in range(10):
                            if time.time() - start_time > TIME_LIMIT - 2.0 or calls >= max_calls - 2:
                                break
                            fn = _f(nx)
                            if math.isnan(fn):
                                break
                            if abs(fn) <= 1e-8:
                                return nx
                            new_x = nx - fn / df
                            if not math.isfinite(new_x):
                                break
                            nx = new_x
                            # Update df?
                            h = max(1e-8 * abs(nx), 1e-12)
                            fn_plus = _f(nx + h)
                            fn_minus = _f(nx - h)
                            if not math.isnan(fn_plus) and not math.isnan(fn_minus):
                                df = (fn_plus - fn_minus) / (2 * h)
                            else:
                                break
                else:
                    # One-sided
                    if not math.isnan(fn_plus):
                        df = (fn_plus - fn) / h
                    elif not math.isnan(fn_minus):
                        df = (fn - fn_minus) / h
                    else:
                        df = 0
                    if abs(df) > 1e-30:
                        for _ in range(10):
                            if time.time() - start_time > TIME_LIMIT - 2.0 or calls >= max_calls - 2:
                                break
                            fn = _f(nx)
                            if math.isnan(fn):
                                break
                            if abs(fn) <= 1e-8:
                                return nx
                            new_x = nx - fn / df
                            if not math.isfinite(new_x):
                                break
                            nx = new_x
                            h = max(1e-8 * abs(nx), 1e-12)
                            fn_plus = _f(nx + h)
                            fn_minus = _f(nx - h)
                            if not math.isnan(fn_plus) and not math.isnan(fn_minus):
                                df = (fn_plus - fn_minus) / (2 * h)
                            elif not math.isnan(fn_plus):
                                df = (fn_plus - fn) / h
                            elif not math.isnan(fn_minus):
                                df = (fn - fn_minus) / h
                            else:
                                break
            if time.time() - start_time < TIME_LIMIT - 1.0 and calls < max_calls:
                f_final = _f(nx)
                if not math.isnan(f_final) and abs(f_final) <= 1e-8:
                    return nx
                if not math.isnan(f_final) and abs(f_final) < best_abs_f:
                    best_x = nx
                    best_abs_f = abs(f_final)

    final_f = _f(best_x)
    if not math.isnan(final_f) and abs(final_f) <= 1e-8:
        return best_x

    return best_x
