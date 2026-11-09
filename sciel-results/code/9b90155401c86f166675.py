"""Root finding for ill-conditioned equations: Robust Adaptive Hybrid."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin below 120s limit
    MAX_EVALS = 490   # Safety margin below 500 limit
    
    # State tracking
    eval_count = 0
    best_x = float(x0)
    best_f = None
    
    def check_budget():
        nonlocal eval_count
        if eval_count > MAX_EVALS:
            return True
        if time.time() - start_time > DEADLINE:
            return True
        return False
    
    def safe_eval(x):
        nonlocal eval_count, best_x, best_f
        if check_budget():
            return None
        try:
            fx = f(x)
            eval_count += 1
            if fx is not None and isinstance(fx, (int, float)) and math.isfinite(fx):
                if best_f is None or abs(fx) < abs(best_f):
                    best_x = x
                    best_f = fx
                return fx
            else:
                eval_count += 1
                return None
        except (OverflowError, ZeroDivisionError, ValueError, TypeError):
            eval_count += 1
            return None
    
    def newton_step(x, fx, h_base=1e-7):
        """Try a damped Newton step with adaptive step size."""
        nonlocal eval_count
        # Compute derivative using central difference
        h = h_base * max(1.0, abs(x))
        fx_plus = safe_eval(x + h)
        if fx_plus is None:
            return None
        fx_minus = safe_eval(x - h)
        if fx_minus is None:
            return None
        
        d = (fx_plus - fx_minus) / (2 * h)
        if abs(d) < 1e-15:
            return None
        
        # Damped Newton: try full step first, then halve
        step = fx / d
        for damp in [1.0, 0.5, 0.25, 0.1, 0.01]:
            x_new = x - damp * step
            fx_new = safe_eval(x_new)
            if fx_new is None:
                continue
            # Accept if it reduces |f|
            if abs(fx_new) < abs(fx):
                return x_new, fx_new
        return None
    
    def bisection_step(a, fa, b, fb):
        """One bisection step."""
        mid = (a + b) / 2.0
        fm = safe_eval(mid)
        if fm is None:
            return None
        if abs(fm) <= 1e-8:
            return mid, fm, True
        if fa * fm < 0:
            return a, fa, mid, fm, False
        else:
            return mid, fm, b, fb, False
    
    # Phase 1: Try Newton's method from x0
    x = float(x0)
    fx = safe_eval(x)
    if fx is not None and abs(fx) <= 1e-8:
        return x
    
    if fx is not None:
        for _ in range(100):
            if check_budget():
                break
            result = newton_step(x, fx)
            if result is None:
                break
            x, fx = result
            if abs(fx) <= 1e-8:
                return x
    
    if best_f is not None and abs(best_f) <= 1e-8:
        return best_x
    
    # Phase 2: Try to bracket the root using expanding intervals
    # Start from x0 and expand in both directions
    def try_bracket(center, fcenter):
        """Try to find a bracket around center."""
        if fcenter is None:
            return None
        if abs(fcenter) <= 1e-8:
            return center, fcenter, center, fcenter
        
        step = max(1.0, abs(center)) * 0.1
        for i in range(1, 20):
            if check_budget():
                return None
            s = step * (2 ** i)
            a = center - s
            b = center + s
            fa = safe_eval(a)
            if fa is None:
                continue
            if abs(fa) <= 1e-8:
                return a, fa, a, fa
            fb = safe_eval(b)
            if fb is None:
                continue
            if abs(fb) <= 1e-8:
                return b, fb, b, fb
            if fa * fb < 0:
                return a, fa, b, fb
        return None
    
    bracket = try_bracket(x, fx)
    if bracket is not None:
        if len(bracket) == 2:
            # Already found root
            return bracket[0]
        a, fa, b, fb = bracket
        # Phase 3: Brent's method via bisection + secant hybrid
        # Implement a simple robust root finder
        for _ in range(100):
            if check_budget():
                break
            # Try secant step
            if abs(fb - fa) > 1e-15:
                x_sec = b - fb * (b - a) / (fb - fa)
                if a < x_sec < b:
                    fx_sec = safe_eval(x_sec)
                    if fx_sec is not None:
                        if abs(fx_sec) <= 1e-8:
                            return x_sec
                        if fa * fx_sec < 0:
                            b, fb = x_sec, fx_sec
                        else:
                            a, fa = x_sec, fx_sec
                        continue
            
            # Bisection step
            mid = (a + b) / 2.0
            fm = safe_eval(mid)
            if fm is None:
                break
            if abs(fm) <= 1e-8:
                return mid
            if fa * fm < 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
    
    # Phase 4: If we haven't found it, try more aggressive search
    # Try multiple starting points
    offsets = [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0]
    for offset in offsets:
        if check_budget():
            break
        x_try = float(x0) + offset
        fx_try = safe_eval(x_try)
        if fx_try is None:
            continue
        if abs(fx_try) <= 1e-8:
            return x_try
        
        # Try Newton from this point
        x_new = x_try
        fx_new = fx_try
        for _ in range(50):
            if check_budget():
                break
            result = newton_step(x_new, fx_new)
            if result is None:
                break
            x_new, fx_new = result
            if abs(fx_new) <= 1e-8:
                return x_new
    
    # Return the best we found
    return best_x
# EVOLVE-BLOCK-END
