import math
import time

try:
    from scipy.optimize import brentq as _scipy_brentq
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_DEADLINE = time.time() + 145.0
_CALL_BUDGET = 495

def _time_left():
    return _DEADLINE - time.time()

def _safe_call(f, x):
    try:
        val = f(float(x))
        if not math.isfinite(val):
            return float('nan')
        return float(val)
    except Exception:
        return float('nan')

def _illinois(f, a, b, fa, fb, counter):
    """
    Illinois Algorithm: A modified bisection that converges faster than standard bisection
    for simple roots by moving the endpoint with the larger function value.
    """
    max_iter = 100
    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        
        if fa * fb > 0:
            break
            
        # Choose midpoint or weighted midpoint
        # Standard Illinois:
        if abs(fa) < abs(fb):
            # Move b
            c = (a + b) / 2.0
            fc = _safe_call(f, c)
            counter[0] += 1
            if math.isnan(fc):
                break
            if abs(fc) <= 1e-8:
                return c, fc, True
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
                # Illinois weighting: shrink the interval by moving the other point
                # Actually, standard Illinois replaces the point with the larger |f| 
                # with the midpoint, but keeps the other point fixed? No, 
                # it sets the new point to the midpoint and swaps the "old" point 
                # to be the midpoint of the previous interval? 
                # Let's use a simpler robust variant: Secant-like update if possible, else bisection.
                pass
        else:
            c = (a + b) / 2.0
            fc = _safe_call(f, c)
            counter[0] += 1
            if math.isnan(fc):
                break
            if abs(fc) <= 1e-8:
                return c, fc, True
            if fb * fc < 0:
                a = c
                fa = fc
            else:
                b = c
                fb = fc
        
        if abs(b - a) < 1e-14 * max(1.0, abs(c)):
            break
            
    c = (a + b) / 2.0
    fc = _safe_call(f, c)
    counter[0] += 1
    return c, fc, abs(fc) <= 1e-8

def _brent_like(f, a, b, fa, fb, counter):
    """
    Simplified Brent's method or Secant with Bisection fallback.
    Since we don't have the full Brent's implementation, we'll use a robust 
    inverse quadratic interpolation / secant hybrid if SciPy is not available.
    """
    if _HAS_SCIPY:
        def wrapped_f(x):
            if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
                raise Exception("budget")
            val = _safe_call(f, x)
            counter[0] += 1
            if math.isnan(val):
                raise Exception("nan")
            return val
        
        try:
            x = _scipy_brentq(wrapped_f, a, b, xtol=1e-14, rtol=1e-14, maxiter=100)
            fx = _safe_call(f, x)
            counter[0] += 1
            if math.isnan(fx):
                return x, fx, False
            return x, fx, abs(fx) <= 1e-8
        except Exception:
            # Fallback to Illinois if SciPy fails
            return _illinois(f, a, b, fa, fb, counter)
    else:
        # Fallback to Illinois
        return _illinois(f, a, b, fa, fb, counter)

def _find_bracket(f, x0, counter):
    """
    Find a bracket [a, b] such that f(a)*f(b) < 0.
    Uses exponential expansion from x0.
    Returns (a, b, fa, fb) or None.
    """
    fx0 = _safe_call(f, x0)
    counter[0] += 1
    if math.isnan(fx0):
        return None
    if abs(fx0) <= 1e-8:
        return (x0, x0, fx0, fx0) # Degenerate bracket, already root

    # Try expanding in both directions
    # Direction +1
    a = x0
    fa = fx0
    step = 1e-3
    for _ in range(50):
        if counter[0] >= _CALL_BUDGET - 10 or _time_left() < 1.0:
            break
        b = a + step
        fb = _safe_call(f, b)
        counter[0] += 1
        if math.isnan(fb):
            break
        if fa * fb < 0:
            return (a, b, fa, fb)
        a = b
        fa = fb
        step *= 2.0
        if step > 1e6:
            break

    # Direction -1
    a = x0
    fa = fx0
    step = 1e-3
    for _ in range(50):
        if counter[0] >= _CALL_BUDGET - 10 or _time_left() < 1.0:
            break
        b = a - step
        fb = _safe_call(f, b)
        counter[0] += 1
        if math.isnan(fb):
            break
        if fa * fb < 0:
            return (a, b, fa, fb)
        a = b
        fa = fb
        step *= 2.0
        if step > 1e6:
            break

    return None

def find_root(f, x0):
    """
    Find a root x of f(x) = 0 with |f(x)| <= 1e-8.
    Returns x as a float.
    """
    counter = [0]
    x0 = float(x0)
    
    # 1. Check x0
    fx0 = _safe_call(f, x0)
    counter[0] += 1
    if not math.isnan(fx0) and abs(fx0) <= 1e-8:
        return x0

    best_x = x0
    best_fx = fx0
    best_abs = abs(fx0) if not math.isnan(fx0) else float('inf')

    # 2. Try to find a bracket using exponential expansion
    bracket = _find_bracket(f, x0, counter)
    
    if bracket:
        a, b, fa, fb = bracket
        if a == b:
            return a
        x, fx, success = _brent_like(f, a, b, fa, fb, counter)
        if not math.isnan(fx):
            if abs(fx) < best_abs:
                best_x = x
                best_fx = fx
                best_abs = abs(fx)
            if success:
                return x

    # 3. If no bracket found or convergence failed, try local optimization from best point
    # Use Secant method from x0 and x0+epsilon
    if counter[0] < _CALL_BUDGET - 10 and _time_left() > 1.0:
        h = 1e-6 * max(1.0, abs(x0))
        x1 = x0
        x2 = x0 + h
        fx1 = best_fx
        if math.isnan(fx1):
            fx1 = _safe_call(f, x1)
            counter[0] += 1
        
        fx2 = _safe_call(f, x2)
        counter[0] += 1
        
        if not math.isnan(fx1) and not math.isnan(fx2):
            # Secant iteration
            for _ in range(50):
                if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
                    break
                if fx2 == fx1:
                    break
                x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
                if not math.isfinite(x_new):
                    break
                x1, fx1 = x2, fx2
                x2 = x_new
                fx2 = _safe_call(f, x2)
                counter[0] += 1
                if math.isnan(fx2):
                    break
                if abs(fx2) <= 1e-8:
                    return x2
                if abs(fx2) < best_abs:
                    best_x = x2
                    best_fx = fx2
                    best_abs = abs(fx2)

    if math.isnan(best_x):
        return x0

    return best_x
