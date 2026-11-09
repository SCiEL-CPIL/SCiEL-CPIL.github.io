"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a robust hybrid approach with Illinois bisection and secant method.
    """
    x0 = float(x0)
    max_evals = 500
    evals = 0
    deadline = time.time() + 110.0
    
    def feval(x):
        nonlocal evals
        if evals >= max_evals:
            return None
        if time.time() > deadline:
            return None
        try:
            val = f(x)
            evals += 1
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                return None
            return val
        except Exception:
            evals += 1
            return None
    
    x = x0
    fx = feval(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-8:
        return x
    
    # Try to find a bracket by checking nearby points
    h = 1e-4 * max(1.0, abs(x0))
    a, fa = x, fx
    b, fb = None, None
    
    for factor in [1, 10, 100, 1000]:
        if time.time() > deadline or evals >= max_evals:
            break
        for sign in [-1, 1]:
            if time.time() > deadline or evals >= max_evals:
                break
            cx = x0 + sign * factor * h
            cf = feval(cx)
            if cf is None:
                continue
            if abs(cf) <= 1e-8:
                return cx
            if fa * cf < 0:
                if cf < 0:
                    a, fa = cx, cf
                else:
                    b, fb = cx, cf
                break
        if b is not None:
            break
    
    best_x, best_f = x, fx
    
    if b is not None:
        # Illinois method on [a, b]
        for _ in range(150):
            if time.time() > deadline or evals >= max_evals:
                break
            if abs(fa) > abs(fb):
                x_new = b - fb * (b - a) / (fb - fa)
            else:
                x_new = a - fa * (b - a) / (fb - fa)
            
            if x_new <= a or x_new >= b:
                x_new = (a + b) / 2.0
            
            fx_new = feval(x_new)
            if fx_new is None:
                break
            if abs(fx_new) < abs(best_f):
                best_x, best_f = x_new, fx_new
            if abs(fx_new) <= 1e-8:
                return x_new
            
            if fa * fx_new < 0:
                b, fb = x_new, fx_new
            else:
                a, fa = x_new, fx_new
                fa *= 0.5
            
            if abs(b - a) < 1e-14 * max(1.0, abs(b)):
                break
    else:
        # No bracket; use secant method
        x1, f1 = x, fx
        x2 = x0 + 1e-4 * max(1.0, abs(x0))
        f2 = feval(x2)
        if f2 is None:
            return best_x
        if abs(f2) < abs(best_f):
            best_x, best_f = x2, f2
        if abs(f2) <= 1e-8:
            return x2
        
        for _ in range(150):
            if time.time() > deadline or evals >= max_evals:
                break
            if abs(f1) < abs(f2):
                x1, f1, x2, f2 = x2, f2, x1, f1
            if abs(f1 - f2) < 1e-15:
                break
            
            x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
            step = abs(x_new - x1)
            max_step = max(1.0, abs(x1))
            if step > max_step:
                x_new = x1 + math.copysign(max_step, x_new - x1)
            
            f_new = feval(x_new)
            if f_new is None:
                break
            if abs(f_new) < abs(best_f):
                best_x, best_f = x_new, f_new
            if abs(f_new) <= 1e-8:
                return x_new
            
            x2, f2 = x1, f1
            x1, f1 = x_new, f_new
    
    # Final polish with central difference
    for _ in range(5):
        if time.time() > deadline or evals >= max_evals:
            break
        h = 1e-7 * max(1.0, abs(best_x))
        fp = feval(best_x + h)
        if fp is None:
            break
        fm = feval(best_x - h)
        if fm is None:
            break
        deriv = (fp - fm) / (2 * h)
        if abs(deriv) < 1e-15:
            break
        # Need f(best_x) - use best_f if it's from best_x
        # Actually we don't have f(best_x) stored reliably, so skip if not available
        # Just return best_x
        break
    
    return best_x
# EVOLVE-BLOCK-END
