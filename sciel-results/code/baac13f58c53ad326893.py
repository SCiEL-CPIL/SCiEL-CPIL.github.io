import math
import time
from scipy.optimize import minimize_scalar

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 108.0
    max_calls = 420
    call_count = 0

    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    def f_call(x):
        nonlocal call_count
        call_count += 1
        return f(x)

    x = float(x0)
    if not math.isfinite(x):
        return x

    fx = f_call(x)

    if not math.isfinite(fx):
        return x

    if abs(fx) <= 1e-8:
        return x

    best_x = x
    best_f = fx

    # Phase 1: Try to find a bracket using geometric expansion
    a = x
    fa = fx
    b = None
    fb = None
    bracket_found = False

    for direction in [1, -1]:
        if not time_ok():
            break
        a = x
        fa = fx
        h = 0.5
        for _ in range(30):
            if not time_ok():
                break
            b = a + direction * h
            fb = f_call(b)
            if not math.isfinite(fb):
                h *= 2.0
                continue
            if abs(fb) < abs(best_f):
                best_x = b
                best_f = fb
            if fa * fb < 0:
                bracket_found = True
                break
            a = b
            fa = fb
            h *= 2.0
        if bracket_found:
            break

    if bracket_found:
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        for _ in range(120):
            if not time_ok():
                break

            if abs(fb) <= 1e-8:
                return b
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) < abs(best_f):
                best_x = b
                best_f = fb
            if abs(fa) < abs(best_f):
                best_x = a
                best_f = fa

            if abs(b - a) < 1e-13 * max(1.0, abs(a), abs(b)):
                if abs(fa) < abs(fb):
                    return a
                else:
                    return b

            if fb != fa:
                x_secant = b - fb * (b - a) / (fb - fa)
                if a < x_secant < b:
                    s = x_secant
                else:
                    s = 0.5 * (a + b)
            else:
                s = 0.5 * (a + b)

            fs = f_call(s)

            if not math.isfinite(fs):
                s = 0.5 * (a + b)
                fs = f_call(s)

            if abs(fs) <= 1e-8:
                return s
            if abs(fs) < abs(best_f):
                best_x = s
                best_f = fs

            if fa * fs < 0:
                b = s
                fb = fs
            else:
                a = s
                fa = fs

        return best_x
    else:
        # No bracket found. Use scipy.optimize.minimize_scalar to minimize f(x)^2
        # This is a structural change: treating root-finding as global optimization
        
        def objective(x_val):
            if not time_ok():
                return float('inf')
            val = f_call(x_val)
            if not math.isfinite(val):
                return float('inf')
            return val * val
        
        # Try multiple starting points for the optimization
        # Use a bounded search around x0
        search_radius = 10.0 * max(1.0, abs(x))
        
        # Try several bounded intervals
        intervals = [
            (x - search_radius, x + search_radius),
            (x - search_radius/2, x + search_radius/2),
            (x - search_radius/4, x + search_radius/4),
        ]
        
        for lo, hi in intervals:
            if not time_ok():
                break
            try:
                result = minimize_scalar(objective, bounds=(lo, hi), method='bounded', 
                                         options={'xatol': 1e-10, 'maxiter': 50})
                if result.success and result.fun <= 1e-16:
                    return result.x
                if abs(math.sqrt(result.fun)) <= 1e-8:
                    return result.x
                if abs(result.fun) < abs(best_f) ** 2:
                    best_x = result.x
                    best_f = math.sqrt(result.fun)
            except Exception:
                pass
        
        # If minimize_scalar didn't find a good root, try Newton-like refinement
        x_curr = best_x
        fx_curr = f_call(x_curr) if abs(fx_curr) > 1e-8 else fx_curr
        
        if not math.isfinite(fx_curr):
            return best_x
            
        h = 1e-6 * max(1.0, abs(x_curr))
        for _ in range(30):
            if not time_ok():
                break
            if abs(fx_curr) <= 1e-8:
                return x_curr
            
            fx_plus = f_call(x_curr + h)
            if not math.isfinite(fx_plus):
                break
            
            d = (fx_plus - fx_curr) / h
            if d == 0 or not math.isfinite(d):
                break
            
            x_new = x_curr - fx_curr / d
            if not math.isfinite(x_new):
                break
            
            step = x_new - x_curr
            max_step = 5.0 * max(1.0, abs(x_curr))
            if abs(step) > max_step:
                x_new = x_curr + max_step * math.copysign(1.0, step)
            
            fx_new = f_call(x_new)
            if not math.isfinite(fx_new):
                break
            
            if abs(fx_new) < abs(best_f):
                best_x = x_new
                best_f = fx_new
            
            if abs(fx_new) < abs(fx_curr):
                x_curr = x_new
                fx_curr = fx_new
            else:
                h *= 0.5
                if h < 1e-12:
                    break
        
        return best_x
