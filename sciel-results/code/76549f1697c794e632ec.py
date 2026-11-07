import math
import time

try:
    from scipy.optimize import brentq as _scipy_brentq
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 5s margin before 150s kill
_CALL_BUDGET = 495
_TOL = 1e-8

def _time_left():
    return _DEADLINE - time.time()

def _safe_call(f, x, counter):
    """
    Safely call f(x), incrementing counter.
    Returns nan on failure.
    """
    if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
        return float('nan')
    try:
        val = f(float(x))
        counter[0] += 1
        if not math.isfinite(val):
            return float('nan')
        return float(val)
    except Exception:
        counter[0] += 1
        return float('nan')

def _illinois_method(f, a, b, fa, fb, counter, max_iter=100):
    """
    Illinois method (modified Regula Falsi) for bracketed root finding.
    Assumes fa and fb have opposite signs.
    """
    if fa * fb > 0:
        return (a + b) / 2.0, _safe_call(f, (a + b) / 2.0, counter)
    
    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        
        # Illinois step: if fa is smaller in magnitude, replace fa with fa/2
        # This forces the interval to shrink faster on the side with the smaller function value.
        if abs(fa) < abs(fb):
            fa = fa / 2.0
            c = (a * fb - b * fa) / (fb - fa)
        else:
            fb = fb / 2.0
            c = (a * fb - b * fa) / (fb - fa)
        
        if not math.isfinite(c) or c <= a or c >= b:
            c = (a + b) / 2.0
        
        fc = _safe_call(f, c, counter)
        if math.isnan(fc):
            break
            
        if abs(fc) <= _TOL:
            return c, fc
            
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
            
        if abs(b - a) < 1e-14 * max(1.0, abs(c)):
            break
            
    # Return the point with the smallest |f| in the final interval
    # We don't want to spend another call if we are out of budget, but we need a valid x.
    # The midpoint is a safe guess, but let's return the one with known smaller f if possible.
    # Actually, we have fa and fb.
    if abs(fa) < abs(fb):
        return a, fa
    else:
        return b, fb

def _find_bracket_local(f, x0, fx0, counter):
    """
    Try to find a bracket close to x0 using small steps.
    Returns (a, b, fa, fb) or None.
    """
    # Try a few small steps in both directions
    steps = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    
    for h in steps:
        # Check x0 + h
        x_plus = x0 + h
        fx_plus = _safe_call(f, x_plus, counter)
        if not math.isnan(fx_plus):
            if fx0 * fx_plus < 0:
                return (x0, x_plus, fx0, fx_plus)
        
        # Check x0 - h
        x_minus = x0 - h
        fx_minus = _safe_call(f, x_minus, counter)
        if not math.isnan(fx_minus):
            if fx0 * fx_minus < 0:
                return (x_minus, x0, fx_minus, fx0)
                
        if counter[0] > _CALL_BUDGET * 0.2: # Don't spend too much on local search
            break

    return None

def _find_bracket_expansion(f, x0, fx0, counter):
    """
    Expand exponentially from x0 to find a bracket.
    """
    best_x = x0
    best_fx = fx0
    best_abs = abs(fx0) if not math.isnan(fx0) else float('inf')
    
    step = 1e-3
    for direction in (1, -1):
        a = x0
        fa = fx0
        for _ in range(50):
            if counter[0] >= _CALL_BUDGET - 10 or _time_left() < 1.0:
                break
            b = a + direction * step
            fb = _safe_call(f, b, counter)
            if math.isnan(fb):
                break
            if fa * fb < 0:
                return (a, b, fa, fb, best_x, best_fx)
            if abs(fb) < best_abs:
                best_x = b
                best_fx = fb
                best_abs = abs(fb)
            a = b
            fa = fb
            step *= 2.0
            if step > 1e6:
                break
    return None

def _secant_fallback(f, x0, fx0, counter):
    """
    Safeguarded Secant method if no bracket is found.
    """
    x1 = x0
    fx1 = fx0
    step = 1e-4 * max(1.0, abs(x0))
    x2 = x0 + step
    fx2 = _safe_call(f, x2, counter)
    
    if math.isnan(fx2):
        return x0, fx0
        
    if abs(fx2) < abs(fx1):
        x1, fx1 = x2, fx2
    else:
        x2, fx2 = x0, fx0

    for _ in range(50):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
            
        if abs(fx1) <= _TOL:
            return x1, fx1
        if abs(fx2) <= _TOL:
            return x2, fx2
            
        if fx1 == fx2:
            break
            
        # Secant step
        denom = fx2 - fx1
        if abs(denom) < 1e-15:
            break
        x_new = x2 - fx2 * (x2 - x1) / denom
        if not math.isfinite(x_new):
            break
            
        # Safeguard: keep x_new close to x2
        if abs(x_new - x2) > 10 * abs(x2 - x1):
            x_new = (x1 + x2) / 2
            
        x1, fx1 = x2, fx2
        x2 = x_new
        fx2 = _safe_call(f, x2, counter)
        if math.isnan(fx2):
            break
            
    if abs(fx1) < abs(fx2):
        return x1, fx1
    return x2, fx2

def find_root(f, x0):
    """
    Find a root x of f(x) = 0 with |f(x)| <= 1e-8.
    Returns x as a float.
    """
    counter = [0]
    x0 = float(x0)
    
    # 1. Check x0
    fx0 = _safe_call(f, x0, counter)
    if not math.isnan(fx0) and abs(fx0) <= _TOL:
        return x0
        
    if math.isnan(fx0):
        # If x0 is invalid, try a nearby point
        x0 = x0 + 1e-5
        fx0 = _safe_call(f, x0, counter)
        if math.isnan(fx0) or abs(fx0) <= _TOL:
            return x0

    best_x = x0
    best_fx = fx0
    best_abs = abs(fx0)

    # 2. Try to find a local bracket
    bracket = _find_bracket_local(f, x0, fx0, counter)
    
    if bracket:
        a, b, fa, fb = bracket
        # Use Brent's method if available, otherwise Illinois
        if _HAS_SCIPY:
            def wrapped_f(x):
                if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
                    raise Exception("budget")
                val = _safe_call(f, x, counter)
                if math.isnan(val):
                    raise Exception("nan")
                return val
            try:
                x = _scipy_brentq(wrapped_f, a, b, xtol=1e-14, rtol=1e-14, maxiter=100)
                fx = _safe_call(f, x, counter)
                if not math.isnan(fx):
                    if abs(fx) <= _TOL:
                        return x
                    if abs(fx) < best_abs:
                        best_x = x
                        best_fx = fx
                        best_abs = abs(fx)
            except Exception:
                pass
        else:
            x, fx = _illinois_method(f, a, b, fa, fb, counter)
            if not math.isnan(fx):
                if abs(fx) <= _TOL:
                    return x
                if abs(fx) < best_abs:
                    best_x = x
                    best_fx = fx
                    best_abs = abs(fx)
    else:
        # 3. If no local bracket, try expansion
        exp_bracket = _find_bracket_expansion(f, x0, fx0, counter)
        if exp_bracket:
            a, b, fa, fb, best_x, best_fx = exp_bracket
            best_abs = abs(best_fx)
            
            if _HAS_SCIPY:
                def wrapped_f(x):
                    if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
                        raise Exception("budget")
                    val = _safe_call(f, x, counter)
                    if math.isnan(val):
                        raise Exception("nan")
                    return val
                try:
                    x = _scipy_brentq(wrapped_f, a, b, xtol=1e-14, rtol=1e-14, maxiter=100)
                    fx = _safe_call(f, x, counter)
                    if not math.isnan(fx):
                        if abs(fx) <= _TOL:
                            return x
                        if abs(fx) < best_abs:
                            best_x = x
                            best_fx = fx
                            best_abs = abs(fx)
                except Exception:
                    pass
            else:
                x, fx = _illinois_method(f, a, b, fa, fb, counter)
                if not math.isnan(fx):
                    if abs(fx) <= _TOL:
                        return x
                    if abs(fx) < best_abs:
                        best_x = x
                        best_fx = fx
                        best_abs = abs(fx)
        else:
            # 4. Fallback to Secant if no bracket found
            x, fx = _secant_fallback(f, x0, fx0, counter)
            if not math.isnan(fx):
                if abs(fx) <= _TOL:
                    return x
                if abs(fx) < best_abs:
                    best_x = x
                    best_fx = fx
                    best_abs = abs(fx)

    return best_x
