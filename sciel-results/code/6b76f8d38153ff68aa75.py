"""Root finding for ill-conditioned equations.

Strategy:
1. Try a fast derivative-free Newton-like method (5-point FD) for a quick win.
2. If that fails, use a robust Illinois algorithm on a bracket if one exists.
3. If no bracket exists, use a robust safeguarded secant method.
4. As a last resort for non-sign-changing roots (e.g., f(x) = (x-1)^2), use a
   local minimization phase to drive |f(x)| down.
5. Track the best result found and enforce a hard deadline.
"""
import math
import time

# EVOLVE-BLOCK-START

_START = None
_DEADLINE = 0.0
_BEST = {"x": None, "abs": float("inf"), "fx": None}
_EVALS = 0
_MAX_EVALS = 490
_TOL = 1e-8

def _elapsed():
    return time.perf_counter() - _START

def _alive():
    return _elapsed() < _DEADLINE and _EVALS < _MAX_EVALS

def _eval(f, x):
    global _EVALS
    if _EVALS >= _MAX_EVALS:
        return None
    _EVALS += 1
    try:
        y = f(x)
        y = float(y)
        if not math.isfinite(y):
            return None
        return y
    except Exception:
        return None

def _record(f, x, fx):
    global _BEST
    if fx is None:
        return
    a = abs(fx)
    if a < _BEST["abs"]:
        _BEST = {"x": x, "fx": fx, "abs": a}

def _finite_diff(f, x, fx, h):
    """5-point central difference for higher accuracy derivative."""
    if fx is None:
        fx = _eval(f, x)
        if fx is None:
            return None
    h = 1e-5 * max(1.0, abs(x))
    
    x_m2 = x - 2*h
    x_m1 = x - h
    x_p1 = x + h
    x_p2 = x + 2*h
    
    f_m2 = _eval(f, x_m2)
    f_m1 = _eval(f, x_m1)
    f_p1 = _eval(f, x_p1)
    f_p2 = _eval(f, x_p2)
    
    if f_m2 is None or f_m1 is None or f_p1 is None or f_p2 is None:
        if f_p1 is None:
            f_p1 = _eval(f, x+h)
        if f_p1 is None:
            return None
        return (f_p1 - fx) / h
        
    numerator = -f_p2 + 8*f_p1 - 8*f_m1 + f_m2
    if numerator == 0.0:
        return 0.0
    return numerator / (12.0 * h)

def _newton(f, x0):
    x = float(x0)
    for _ in range(20):
        if not _alive():
            break
        fx = _eval(f, x)
        if fx is None:
            break
        _record(f, x, fx)
        if abs(fx) <= 1e-12:
            return x
        
        dfx = _finite_diff(f, x, fx, 1.0)
        if dfx is None or abs(dfx) < 1e-15:
            break
        
        step = fx / dfx
        if not math.isfinite(step):
            break
            
        x_new = x - step
        if abs(x_new) > 1000 * max(1.0, abs(x)):
            x_new = x - 0.1 * step
        
        x = x_new
    return x

def _find_bracket(f, x0, fx0):
    if fx0 is None:
        return None
    
    step = 0.1 * max(1.0, abs(x0))
    a = x0
    fa = fx0
    
    for direction in (1.0, -1.0):
        b = x0 + direction * step
        fb = _eval(f, b)
        if fb is None:
            continue
        _record(f, b, fb)
        
        if fa * fb < 0.0:
            if direction > 0:
                return (a, b, fa, fb)
            else:
                return (b, a, fb, fa)
        
        cur = b
        curf = fb
        for _ in range(30):
            if not _alive():
                break
            step *= 2.0
            nxt = x0 + direction * step
            fn = _eval(f, nxt)
            if fn is None:
                break
            _record(f, nxt, fn)
            
            if curf * fn < 0.0:
                if direction > 0:
                    return (cur, nxt, curf, fn)
                else:
                    return (nxt, cur, fn, curf)
            
            cur = nxt
            curf = fn
    return None

def _illinois(f, a, b, fa, fb):
    """Illinois algorithm for robust bracketed root finding."""
    if fa * fb > 0:
        return None
    
    # Ensure |fa| > |fb| so we shrink the side with the larger error
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
        
    c = a
    fc = fa
    
    for _ in range(100):
        if not _alive():
            break
            
        if fb == 0.0:
            return b
        if fa == 0.0:
            return a
            
        # Illinois step:
        # If |fa| > |fb|, we replace fa with fa * (a-c)/(b-c) to force more bisection
        # on the side with the larger error.
        # Actually, standard Illinois:
        # If |fa| > |fb|: fa = fa * (a - c) / (b - c)
        # Else: fb = fb * (b - c) / (a - c)
        # Then use secant step on the modified values.
        
        if abs(fa) > abs(fb):
            fa = fa * (a - c) / (b - c)
        else:
            fb = fb * (b - c) / (a - c)
            
        # Secant step using modified fa, fb
        if abs(fb - fa) < 1e-300:
            s = 0.5 * (a + b)
        else:
            s = (a * fb - b * fa) / (fb - fa)
            
        if not math.isfinite(s) or not (min(a, b) < s < max(a, b)):
            s = 0.5 * (a + b)
            
        fs = _eval(f, s)
        if fs is None:
            break
        _record(f, s, fs)
        
        if fs == 0.0:
            return s
            
        # Update bracket
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
            
        # Keep c as the point we just replaced (the one that didn't change)
        # Wait, c should be the previous point that was NOT updated.
        # In Illinois, c is the point that was replaced in the previous step.
        # Let's stick to the standard loop structure.
        
        # Standard Illinois:
        # c = a
        # fa, fb are current.
        # If |fa| > |fb|:
        #   fa = fa * (a - c) / (b - c)
        #   s = secant(a, fa, b, fb)
        #   if fa * fs < 0: b = s, fb = fs, c = a
        #   else: a = s, fa = fs, c = b
        # Else:
        #   fb = fb * (b - c) / (a - c)
        #   s = secant(a, fa, b, fb)
        #   if fa * fs < 0: b = s, fb = fs, c = a
        #   else: a = s, fa = fs, c = b
        
        # My implementation above didn't update c correctly. Let's redo.
        pass # The logic above was slightly off. Let's rewrite _illinois properly below or inline.
        
    return b

def _illinois_correct(f, a, b, fa, fb):
    if fa * fb > 0:
        return None
        
    c = a
    fc = fa
    
    for _ in range(100):
        if not _alive():
            break
            
        if fb == 0.0: return b
        if fa == 0.0: return a
            
        if abs(fa) > abs(fb):
            fa = fa * (a - c) / (b - c)
        else:
            fb = fb * (b - c) / (a - c)
            
        if abs(fb - fa) < 1e-300:
            s = 0.5 * (a + b)
        else:
            s = (a * fb - b * fa) / (fb - fa)
            
        if not math.isfinite(s) or not (min(a, b) < s < max(a, b)):
            s = 0.5 * (a + b)
            
        fs = _eval(f, s)
        if fs is None:
            break
        _record(f, s, fs)
        
        if fs == 0.0:
            return s
            
        if fa * fs < 0:
            c = a
            b = s
            fb = fs
        else:
            c = b
            a = s
            fa = fs
            
    return b

def _safeguarded_secant(f, x0, fx0):
    """Secant method with bisection safeguard if the step goes out of bounds or diverges."""
    if fx0 is None:
        fx0 = _eval(f, x0)
        if fx0 is None:
            return None
    _record(f, x0, fx0)
    
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h
    fx1 = _eval(f, x1)
    if fx1 is None:
        return None
    _record(f, x1, fx1)
    
    # We don't have a bracket, so we can't strictly safeguard with bisection.
    # But we can limit the step size.
    x_prev = x0
    fx_prev = fx0
    x_curr = x1
    fx_curr = fx1
    
    for _ in range(100):
        if not _alive():
            break
            
        if abs(fx_curr) <= 1e-12:
            return x_curr
            
        if abs(fx_curr - fx_prev) < 1e-300:
            break
            
        x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        
        if not math.isfinite(x_new):
            break
            
        # Limit step size to prevent divergence
        max_step = 10.0 * max(1.0, abs(x_curr))
        if abs(x_new - x_curr) > max_step:
            # Take a smaller step in the direction of x_new
            x_new = x_curr + math.copysign(max_step, x_new - x_curr)
            
        fx_new = _eval(f, x_new)
        if fx_new is None:
            break
        _record(f, x_new, fx_new)
        
        x_prev, fx_prev = x_curr, fx_curr
        x_curr, fx_curr = x_new, fx_new
        
    return x_curr

def _minimize(f, x0):
    """
    Simple local minimization of |f(x)| using a golden section or similar.
    This is a fallback for cases where the root is a local minimum of |f(x)|
    that is not a sign change (e.g. (x-1)^2).
    """
    a = x0 - 0.1 * max(1.0, abs(x0))
    b = x0 + 0.1 * max(1.0, abs(x0))
    
    fa = abs(_eval(f, a) or float('inf'))
    fb = abs(_eval(f, b) or float('inf'))
    
    if fa == float('inf') or fb == float('inf'):
        return None
        
    # Expand if necessary to find a valley
    for _ in range(10):
        if not _alive():
            break
        if fa < fb:
            b = a
            fb = fa
            a = a - (b - a)
            fa = abs(_eval(f, a) or float('inf'))
        else:
            a = b
            fa = fb
            b = b + (b - a)
            fb = abs(_eval(f, b) or float('inf'))
            
        if fa < 1e-12 or fb < 1e-12:
            break
            
        if fa == float('inf') or fb == float('inf'):
            break

    if fa == float('inf') or fb == float('inf'):
        return None

    # Golden section search
    gr = (math.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = abs(_eval(f, c) or float('inf'))
    fd = abs(_eval(f, d) or float('inf'))
    
    for _ in range(50):
        if not _alive():
            break
            
        if fc < fd:
            b = d
            fb = fd
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = abs(_eval(f, c) or float('inf'))
        else:
            a = c
            fa = fc
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = abs(_eval(f, d) or float('inf'))
            
        if abs(b - a) < 1e-12:
            break
            
    x_mid = 0.5 * (a + b)
    return x_mid

def _polish(f, x, fx):
    """
    Given a candidate x with small fx, try a final secant step with a tight tolerance
    to ensure we are within 1e-8.
    """
    if fx is None:
        fx = _eval(f, x)
        if fx is None:
            return x, None
    if abs(fx) <= 1e-8:
        return x, fx
    
    # Try a small perturbation secant
    h = 1e-5 * max(1.0, abs(x))
    x2 = x + h
    f2 = _eval(f, x2)
    if f2 is None:
        return x, fx
        
    _record(f, x2, f2)
    
    if abs(fx - f2) < 1e-15:
        return x, fx
        
    x_new = x - fx * (x - x2) / (fx - f2)
    
    if math.isfinite(x_new):
        f_new = _eval(f, x_new)
        if f_new is not None:
            _record(f, x_new, f_new)
            if abs(f_new) < abs(fx):
                return x_new, f_new
            if abs(f_new) <= 1e-8:
                return x_new, f_new
    return x, fx

def find_root(f, x0):
    global _START, _DEADLINE, _BEST, _EVALS
    _START = time.perf_counter()
    _DEADLINE = _START + 110.0
    _EVALS = 0
    _BEST = {"x": float(x0), "abs": float("inf"), "fx": None}
    
    x0f = float(x0)
    fx0 = _eval(f, x0f)
    _record(f, x0f, fx0)
    
    if fx0 is not None and abs(fx0) <= _TOL:
        return x0f

    # 1. Newton (with FD derivative)
    x_n = _newton(f, x0f)
    if x_n is not None:
        f_n = _eval(f, x_n)
        if f_n is not None:
            _record(f, x_n, f_n)
            if abs(f_n) <= _TOL:
                x_p, f_p = _polish(f, x_n, f_n)
                if f_p is not None and abs(f_p) <= _TOL:
                    return x_p
                return x_n

    # 2. Find Bracket and use Illinois
    bracket = _find_bracket(f, x0f, fx0)
    if bracket is not None:
        a, b, fa, fb = bracket
        x_b = _illinois_correct(f, a, b, fa, fb)
        if x_b is not None:
            f_b = _eval(f, x_b)
            if f_b is not None:
                _record(f, x_b, f_b)
                if abs(f_b) <= _TOL:
                    x_p, f_p = _polish(f, x_b, f_b)
                    if f_p is not None and abs(f_p) <= _TOL:
                        return x_p
                    return x_b

    # 3. Safeguarded Secant
    x_s = _safeguarded_secant(f, x0f, fx0)
    if x_s is not None:
        f_s = _eval(f, x_s)
        if f_s is not None:
            _record(f, x_s, f_s)
            if abs(f_s) <= _TOL:
                x_p, f_p = _polish(f, x_s, f_s)
                if f_p is not None and abs(f_p) <= _TOL:
                    return x_p
                return x_s

    # 4. Minimization fallback
    x_m = _minimize(f, x0f)
    if x_m is not None:
        f_m = _eval(f, x_m)
        if f_m is not None:
            _record(f, x_m, f_m)
            if abs(f_m) <= _TOL:
                x_p, f_p = _polish(f, x_m, f_m)
                if f_p is not None and abs(f_p) <= _TOL:
                    return x_p
                return x_m

    # 5. Fallback: Return best found
    if _BEST["fx"] is not None and abs(_BEST["fx"]) <= _TOL:
        return _BEST["x"]
        
    return _BEST["x"]

# EVOLVE-BLOCK-END
