import math
import time

def find_root(f, x0):
    START_TIME = time.time()
    TIME_LIMIT = 145.0
    MAX_CALLS = 495
    calls = 0
    cache = {}
    best_x = x0
    best_f = float('inf')
    found = False

    def safe_f(x):
        nonlocal calls, best_x, best_f, found
        if found:
            return 0.0
        if calls >= MAX_CALLS:
            return float('nan')
        if time.time() - START_TIME > TIME_LIMIT:
            return float('nan')
        
        if x in cache:
            val = cache[x]
        else:
            calls += 1
            try:
                val = f(x)
                val = float(val)
            except Exception:
                val = float('nan')
            cache[x] = val
            
        if math.isfinite(val):
            abs_val = abs(val)
            if abs_val < best_f:
                best_f = abs_val
                best_x = x
                if best_f <= 1e-8:
                    found = True
        return val

    def is_bracket(xa, xb, fa, fb):
        if not math.isfinite(fa) or not math.isfinite(fb):
            return False
        return fa * fb < 0

    def refine_root(a, b, fa, fb):
        nonlocal best_x, best_f, found
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        # Steffensen's method for faster convergence (quadratic, no derivative needed)
        # Start with secant/bisection hybrid to ensure bracket maintenance
        x0_s = (a + b) / 2.0
        x1_s = a
        f1_s = fa
        
        for _ in range(30):
            if found:
                return best_x
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            # Steffensen step: x_{n+1} = x_n - f(x_n)^2 / (f(x_n + f(x_n)) - f(x_n))
            # But we need to be careful about the step size
            # Use a modified Steffensen that stays in bracket
            
            # First try secant step
            if fa != fb and (b - a) > 1e-14:
                x_sec = a - fa * (b - a) / (fb - fa)
                if x_sec > a and x_sec < b:
                    f_sec = safe_f(x_sec)
                    if math.isfinite(f_sec):
                        if abs(f_sec) <= 1e-8:
                            return x_sec
                        if fa * f_sec < 0:
                            b = x_sec
                            fb = f_sec
                        else:
                            a = x_sec
                            fa = f_sec
                        continue
            
            # Bisection fallback
            mid = (a + b) / 2.0
            fm = safe_f(mid)
            if not math.isfinite(fm):
                break
            
            if abs(fm) <= 1e-8:
                return mid

            if fa * fm < 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
        
        # Final polish with more iterations if bracket is small
        if b - a < 1e-6:
            for _ in range(20):
                if found:
                    return best_x
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                if fa != fb and (b - a) > 1e-14:
                    x_sec = a - fa * (b - a) / (fb - fa)
                    if x_sec > a and x_sec < b:
                        f_sec = safe_f(x_sec)
                        if math.isfinite(f_sec):
                            if abs(f_sec) <= 1e-8:
                                return x_sec
                            if fa * f_sec < 0:
                                b = x_sec
                                fb = f_sec
                            else:
                                a = x_sec
                                fa = f_sec
                            continue
                mid = (a + b) / 2.0
                fm = safe_f(mid)
                if not math.isfinite(fm):
                    break
                if abs(fm) <= 1e-8:
                    return mid
                if fa * fm < 0:
                    b = mid
                    fb = fm
                else:
                    a = mid
                    fa = fm
        
        # Return best candidate
        candidates = [a, b]
        if 'mid' in locals():
            candidates.append(mid)
        if 'x_sec' in locals():
            candidates.append(x_sec)
        
        best_local = min(candidates, key=lambda x: abs(cache.get(x, safe_f(x))))
        return best_local

    # Evaluate initial point
    fx0 = safe_f(x0)
    if found:
        return best_x

    # Dense uniform grid scan to find sign changes quickly
    # Use a moderate grid that covers likely root locations
    
    # Try multiple grid scales to handle different root locations
    grid_scales = [0.1, 0.5, 1.0, 5.0, 10.0]
    
    for scale in grid_scales:
        if found:
            return best_x
        if time.time() - START_TIME > TIME_LIMIT:
            break
        if calls >= MAX_CALLS - 5:
            break
        
        # Scan in both directions with uniform spacing
        for direction in [1, -1]:
            if found:
                return best_x
            if time.time() - START_TIME > TIME_LIMIT:
                break
            if calls >= MAX_CALLS - 5:
                break
            
            # Number of points to scan in this direction
            n_points = min(20, int((MAX_CALLS - calls) / 2))
            if n_points < 2:
                continue
            
            x_prev = x0
            fx_prev = fx0
            
            for i in range(1, n_points + 1):
                if found:
                    return best_x
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                if calls >= MAX_CALLS - 5:
                    break
                
                x_curr = x0 + direction * i * scale
                fx_curr = safe_f(x_curr)
                
                if found:
                    return best_x
                
                if is_bracket(x_prev, x_curr, fx_prev, fx_curr):
                    return refine_root(x_prev, x_curr, fx_prev, fx_curr)
                
                x_prev = x_curr
                fx_prev = fx_curr

    # If no bracket found with uniform grid, try geometric expansion
    if not found:
        base_step = 0.1
        expansion = 2.0
        max_range = 10000.0
        
        for direction in [1, -1]:
            if found:
                return best_x
            if time.time() - START_TIME > TIME_LIMIT:
                break
            if calls >= MAX_CALLS - 10:
                break
            
            step = base_step
            x_prev = x0
            fx_prev = fx0
            x_curr = x0 + direction * step
            fx_curr = safe_f(x_curr)
            
            if found:
                return best_x
            
            if is_bracket(x_prev, x_curr, fx_prev, fx_curr):
                return refine_root(x_prev, x_curr, fx_prev, fx_curr)
            
            # Geometric expansion
            while abs(x_curr - x0) < max_range and calls < MAX_CALLS - 10:
                if time.time() - START_TIME > TIME_LIMIT:
                    break
                
                x_prev = x_curr
                fx_prev = fx_curr
                step *= expansion
                x_curr = x0 + direction * step
                fx_curr = safe_f(x_curr)
                
                if found:
                    return best_x
                
                if is_bracket(x_prev, x_curr, fx_prev, fx_curr):
                    return refine_root(x_prev, x_curr, fx_prev, fx_curr)
                
                # If we've gone too far, try a finer step between last two points
                if abs(x_curr - x0) > 100:
                    # Do a finer scan between x_prev and x_curr
                    n_fine = 10
                    for i in range(1, n_fine + 1):
                        if calls >= MAX_CALLS - 5:
                            break
                        x_fine = x_prev + (x_curr - x_prev) * i / n_fine
                        fx_fine = safe_f(x_fine)
                        if found:
                            return best_x
                        if is_bracket(x_prev, x_fine, fx_prev, fx_fine):
                            return refine_root(x_prev, x_fine, fx_prev, fx_fine)
                        x_prev = x_fine
                        fx_prev = fx_fine
                    
                    # Reset to continue from x_curr
                    x_prev = x_curr
                    fx_prev = fx_curr
                    step = base_step
                    x_curr = x0 + direction * step
                    fx_curr = safe_f(x_curr)
                    if found:
                        return best_x
                    if is_bracket(x_prev, x_curr, fx_prev, fx_curr):
                        return refine_root(x_prev, x_curr, fx_prev, fx_curr)

    return best_x
