import math
import time

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations of f.
    Strategy: wide deterministic bracketing sweep to find a sign-change interval,
    then Illinois/inverse-interpolation hybrid to converge. No derivative estimates.
    """
    start_time = time.time()
    MAX_WALL = 115.0
    MAX_EVALS = 500
    TARGET_TOL = 1e-8

    eval_count = [0]
    cache = {}
    best_x = float(x0)
    best_f = float('inf')

    def check_time():
        return (time.time() - start_time) < MAX_WALL

    def eval_f(x):
        nonlocal best_x, best_f
        if abs(x) < 1e15:
            key = round(x, 14)
        else:
            key = x
        if key in cache:
            val = cache[key]
        else:
            eval_count[0] += 1
            try:
                val = f(x)
            except Exception:
                val = 0.0
            if not math.isfinite(val):
                val = 0.0
            cache[key] = val
        if abs(val) < abs(best_f):
            best_x = x
            best_f = val
        return val

    def is_good(x, fx=None):
        if fx is None:
            fx = eval_f(x)
        return abs(fx) <= TARGET_TOL

    # Phase 1: Wide deterministic bracketing sweep
    def find_bracket(x0):
        """
        Search outward from x0 in both directions with geometrically increasing steps
        to find a sign-change bracket [a, b]. Returns (a, b, fa, fb) or None.
        """
        x = float(x0)
        fx = eval_f(x)
        if abs(fx) <= TARGET_TOL:
            return (x, x, fx, fx)

        # Try a set of step sizes, both directions
        # Start with small steps relative to x0, grow geometrically
        base_step = max(1e-3, abs(x) * 1e-3)
        
        # First: try very close points to see if we're already near a root
        for h in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
            if not check_time() or eval_count[0] >= MAX_EVALS - 50:
                break
            for d in [1, -1]:
                x_try = x + d * h
                fx_try = eval_f(x_try)
                if abs(fx_try) <= TARGET_TOL:
                    return (x_try, x_try, fx_try, fx_try)
                if fx * fx_try < 0:
                    if d == 1:
                        return (x, x_try, fx, fx_try)
                    else:
                        return (x_try, x, fx_try, fx)
        
        # Now sweep outward
        step = base_step
        for direction in [1, -1]:
            for i in range(60):
                if not check_time() or eval_count[0] >= MAX_EVALS - 30:
                    break
                x_new = x + direction * step
                fx_new = eval_f(x_new)
                if abs(fx_new) <= TARGET_TOL:
                    return (x_new, x_new, fx_new, fx_new)
                if fx * fx_new < 0:
                    if direction == 1:
                        return (x, x_new, fx, fx_new)
                    else:
                        return (x_new, x, fx_new, fx)
                step *= 2.0
                if step > 1e12:
                    break
        
        return None

    # Phase 2: Illinois / inverse-interpolation hybrid
    def illinois(a, b, fa, fb, max_iters=200):
        """
        Illinois algorithm: a variant of regula falsi that shrinks the bracket
        on the side that has been updated more recently. Guaranteed to converge.
        """
        nonlocal best_x, best_f
        
        # Ensure |fa| <= |fb| so we update b (the side with larger |f|)
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        for i in range(max_iters):
            if not check_time() or eval_count[0] >= MAX_EVALS - 5:
                break
            
            # Check endpoints
            if abs(fa) <= TARGET_TOL:
                best_x = a
                best_f = fa
                return True
            if abs(fb) <= TARGET_TOL:
                best_x = b
                best_f = fb
                return True
            
            # Inverse linear interpolation (regula falsi)
            denom = fb - fa
            if denom == 0:
                # Fallback to bisection
                c = (a + b) / 2.0
            else:
                c = a - fa * (b - a) / denom
            
            # Keep c within [a, b]
            if c <= a or c >= b:
                c = (a + b) / 2.0
            
            fc = eval_f(c)
            
            if abs(fc) <= TARGET_TOL:
                best_x = c
                best_f = fc
                return True
            
            # Illinois: shrink the side we're replacing
            if fa * fc < 0:
                # Root is in [a, c]
                b = c
                fb = fc * 0.5  # Illinois damping
            else:
                # Root is in [c, b]
                a = c
                fa = fc * 0.5  # Illinois damping
            
            # Check convergence
            if abs(b - a) < 1e-15 * max(1.0, abs(c)):
                break
        
        # Return best of a, b, c
        if abs(fa) <= TARGET_TOL:
            best_x = a
            best_f = fa
            return True
        if abs(fb) <= TARGET_TOL:
            best_x = b
            best_f = fb
            return True
        return False

    # Initial evaluation
    x = float(x0)
    fx = eval_f(x)
    if abs(fx) <= TARGET_TOL:
        return x

    # Phase 1: Find bracket
    bracket = find_bracket(x)
    
    if bracket is not None:
        a, b, fa, fb = bracket
        if a != b:
            # Phase 2: Illinois on the bracket
            if illinois(a, b, fa, fb, max_iters=200):
                return best_x
        else:
            # Already found a root
            if abs(fa) <= TARGET_TOL:
                return a

    # If no bracket found, try a secant-like approach as fallback
    # Use the best point and a perturbation
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    f1 = eval_f(x1)
    
    if abs(f1) <= TARGET_TOL:
        return x1
    
    # Try secant iterations from (x, x1)
    x_prev = x
    x_curr = x1
    f_prev = fx
    f_curr = f1
    
    for i in range(100):
        if not check_time() or eval_count[0] >= MAX_EVALS - 5:
            break
        if abs(f_curr) <= TARGET_TOL:
            best_x = x_curr
            best_f = f_curr
            return best_x
        if abs(f_prev) <= TARGET_TOL:
            best_x = x_prev
            best_f = f_prev
            return best_x
        
        denom = f_curr - f_prev
        if denom == 0:
            break
        x_new = x_curr - f_curr * (x_curr - x_prev) / denom
        if not math.isfinite(x_new):
            break
        
        x_prev = x_curr
        x_curr = x_new
        f_prev = f_curr
        f_curr = eval_f(x_curr)
    
    return best_x
