import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0
    max_calls = 480
    call_count = 0
    best_x = float(x0)
    best_f = float('inf')

    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    def f_call(x):
        nonlocal call_count, best_x, best_f
        call_count += 1
        try:
            fx = f(x)
        except Exception:
            fx = float('nan')
        if math.isfinite(fx) and abs(fx) < abs(best_f):
            best_f = fx
            best_x = x
        return fx

    def secant_method(x1, f1, max_iter=50):
        """Secant method: 1 call per iteration, superlinear convergence."""
        nonlocal best_x, best_f
        x0_s = x1
        f0_s = f1
        # Need a second point; use a small perturbation
        h = 1e-4 * max(1.0, abs(x0_s))
        x1_s = x0_s + h
        f1_s = f_call(x1_s)
        if not math.isfinite(f1_s):
            return None
        
        for _ in range(max_iter):
            if not time_ok():
                break
            if abs(f1_s) <= 1e-12:
                return x1_s
            if abs(f0_s) <= 1e-12:
                return x0_s
            
            df = f1_s - f0_s
            if abs(df) < 1e-30:
                break
            
            x_new = x1_s - f1_s * (x1_s - x0_s) / df
            if not math.isfinite(x_new):
                break
            
            # If step is too large, dampen
            if abs(x_new - x1_s) > 100.0 * max(1.0, abs(x1_s)):
                x_new = x1_s + 10.0 * math.copysign(1.0, x_new - x1_s)
            
            f_new = f_call(x_new)
            if not math.isfinite(f_new):
                break
            if abs(f_new) <= 1e-12:
                return x_new
            
            # Shift: x0 <- x1, x1 <- new
            x0_s, f0_s = x1_s, f1_s
            x1_s, f1_s = x_new, f_new
            
        # Return best of the last few points if they meet tolerance
        if math.isfinite(f0_s) and abs(f0_s) <= 1e-8:
            return x0_s
        if math.isfinite(f1_s) and abs(f1_s) <= 1e-8:
            return x1_s
        return None

    def bisection(a, fa, b, fb, max_iter=60):
        """Robust bisection fallback."""
        nonlocal best_x, best_f
        for _ in range(max_iter):
            if not time_ok():
                break
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                return a if abs(fa) < abs(fb) else b
            
            mid = 0.5 * (a + b)
            fm = f_call(mid)
            if not math.isfinite(fm):
                break
            if abs(fm) <= 1e-8:
                return mid
            
            if fa * fm < 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
        
        return a if abs(fa) < abs(fb) else b

    x = float(x0)
    if not math.isfinite(x):
        return best_x

    fx = f_call(x)
    if not math.isfinite(fx):
        return best_x
    if abs(fx) <= 1e-8:
        return x

    # Strategy 1: Try secant method from x0
    result = secant_method(x, fx)
    if result is not None and math.isfinite(result):
        return result

    # Strategy 2: Find a bracket, then use bisection
    a = x
    fa = fx
    h = 1.0
    b = None
    fb = None
    
    for _ in range(25):
        if not time_ok():
            break
        
        b_right = a + h
        fb_right = f_call(b_right)
        if math.isfinite(fb_right) and fa * fb_right < 0:
            b = b_right
            fb = fb_right
            break
        
        b_left = a - h
        fb_left = f_call(b_left)
        if math.isfinite(fb_left) and fa * fb_left < 0:
            b = b_left
            fb = fb_left
            break
        
        # Move to the point with smaller |f|
        if math.isfinite(fb_right) and abs(fb_right) < abs(fa):
            a = b_right
            fa = fb_right
        elif math.isfinite(fb_left) and abs(fb_left) < abs(fa):
            a = b_left
            fa = fb_left
        
        h *= 2.0

    if b is not None:
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        result = bisection(a, fa, b, fb)
        if result is not None and math.isfinite(result):
            return result

    # Strategy 3: Try secant from various perturbed points
    for pert in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]:
        if not time_ok():
            break
        x_try = x0 + pert * max(1.0, abs(x0))
        if not math.isfinite(x_try):
            continue
        fx_try = f_call(x_try)
        if not math.isfinite(fx_try):
            continue
        if abs(fx_try) <= 1e-8:
            return x_try
        result = secant_method(x_try, fx_try, max_iter=30)
        if result is not None and math.isfinite(result):
            return result

    # Final: return best found
    return best_x
