import numpy as np
import math
import time

def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses cubic polynomial reconstruction for speed.
    """
    DEADLINE = 110.0
    START_TIME = time.time()
    
    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    def is_finite(x):
        return math.isfinite(x)

    def safe_f(x):
        try:
            fx = f(x)
            if is_finite(fx):
                return fx
        except:
            pass
        return None

    x = float(x0)
    
    # 1. Check initial point
    fx0 = safe_f(x)
    if fx0 is None:
        return x
    if abs(fx0) <= 1e-8:
        return x

    # 2. Cubic Reconstruction
    # Choose 4 points. Scale step size based on x0 to handle large/small scales?
    # If x0 is 0, step 1. If x0 is 1e6, step 1e6?
    # Polynomials are smooth, but roots are fixed.
    # Let's use a relative step.
    scale = max(1.0, abs(x))
    h = 1.0 * scale
    
    # Points: x-h, x, x+h, x+2h
    # We already have fx0 at x.
    x1 = x - h
    x2 = x + h
    x3 = x + 2*h
    
    fx1 = safe_f(x1)
    if fx1 is None: return x
    fx2 = safe_f(x2)
    if fx2 is None: return x
    fx3 = safe_f(x3)
    if fx3 is None: return x

    # Fit cubic
    # Coefficients for P(t) where t = (x - x0) / h ? No, just fit in x.
    # np.polyfit returns coefficients [a3, a2, a1, a0] for a3*x^3 + ...
    try:
        coeffs = np.polyfit([x1, x, x2, x3], [fx1, fx0, fx2, fx3], 3)
        roots = np.roots(coeffs)
    except Exception:
        # Fallback to Secant
        return _secant_fallback(f, x, fx0, check_deadline, safe_f)

    best_x = x
    best_fx = fx0
    
    for r in roots:
        if not np.isreal(r):
            continue
        r_real = float(np.real(r))
        if not is_finite(r_real):
            continue
        fr = safe_f(r_real)
        if fr is None:
            continue
        if abs(fr) < abs(best_fx):
            best_fx = fr
            best_x = r_real
        if abs(fr) <= 1e-8:
            return r_real

    # If cubic didn't find a good root, fallback
    if abs(best_fx) <= 1e-8:
        return best_x
        
    # Try a second cubic with wider spread if we have time
    if check_deadline():
        h2 = 10.0 * scale
        x4 = x - h2
        x5 = x + h2
        x6 = x + 2*h2
        
        fx4 = safe_f(x4)
        if fx4 is not None:
            fx5 = safe_f(x5)
            if fx5 is not None:
                fx6 = safe_f(x6)
                if fx6 is not None:
                    try:
                        coeffs2 = np.polyfit([x4, x, x5, x6], [fx4, fx0, fx5, fx6], 3)
                        roots2 = np.roots(coeffs2)
                        for r in roots2:
                            if not np.isreal(r):
                                continue
                            r_real = float(np.real(r))
                            if not is_finite(r_real):
                                continue
                            fr = safe_f(r_real)
                            if fr is None:
                                continue
                            if abs(fr) <= 1e-8:
                                return r_real
                            if abs(fr) < abs(best_fx):
                                best_fx = fr
                                best_x = r_real
                    except Exception:
                        pass

    return _secant_fallback(f, best_x, best_fx, check_deadline, safe_f)

def _secant_fallback(f, x_start, fx_start, check_deadline, safe_f):
    """
    Simple Secant method fallback.
    """
    if fx_start is None:
        return x_start
    if abs(fx_start) <= 1e-8:
        return x_start

    x_prev = x_start
    fx_prev = fx_start
    h = 1e-4 * max(1.0, abs(x_start))
    x_curr = x_start + h
    fx_curr = safe_f(x_curr)
    
    if fx_curr is None:
        return x_start

    best_x = x_curr if abs(fx_curr) < abs(fx_prev) else x_prev
    best_fx = fx_curr if abs(fx_curr) < abs(fx_prev) else fx_prev

    for _ in range(20):
        if not check_deadline():
            break
        if abs(fx_curr) <= 1e-8:
            return x_curr
        if abs(fx_prev) <= 1e-8:
            return x_prev

        if fx_prev == fx_curr:
            break

        denom = fx_curr - fx_prev
        if denom == 0:
            break

        x_next = x_curr - fx_curr * (x_curr - x_prev) / denom
        if not math.isfinite(x_next):
            break

        fx_next = safe_f(x_next)
        if fx_next is None:
            break

        if abs(fx_next) < abs(best_fx):
            best_x = x_next
            best_fx = fx_next

        x_prev = x_curr
        fx_prev = fx_curr
        x_curr = x_next
        fx_curr = fx_next

    return best_x
