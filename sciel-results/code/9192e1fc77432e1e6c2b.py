"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START

def _safe_f(f, x):
    """Evaluate f(x) safely, returning inf on any error or non-finite result."""
    try:
        v = f(float(x))
        v = float(v)
        if not math.isfinite(v):
            return float('inf')
        return v
    except Exception:
        return float('inf')


def _brent(f, a, b, fa, fb, tol=1e-12, max_iter=100):
    """
    Brent's method for root finding.
    Returns (root, converged).
    """
    if abs(fa) <= tol:
        return a, True
    if abs(fb) <= tol:
        return b, True
    
    c = a
    fc = fa
    d = b - a
    e = d
    
    for _ in range(max_iter):
        if abs(fb) <= tol:
            return b, True
        if fb * fc > 0:
            c = a
            fc = fa
            d = b - c
            e = d
        if abs(e) < 1e-14:
            d = b - c
            e = d / 2.0
        else:
            # Inverse quadratic interpolation
            t1 = 2.0 * fc
            tmp = (b - c) * (fb - fc)
            t2 = (b - a) * (fb - fa)
            s = (c - b) * t2 * t1
            # Check if quadratic is valid
            if abs(s) > abs(t2 * t2):
                s /= t2
            else:
                s = e
                e = d
                d = b - c
                continue
            # Check bounds
            if (s < 3.0 * e / 2.0 and s < e) or (s > e and e < 0) or (s < e and e > 0):
                s = e
                e = d
                d = b - c
                continue
            if s == 0:
                s = e
                e = d
                d = b - c
                continue
            d = b - c
            e = s
            b = c + s
            fb = _safe_f(f, b)
            if math.isinf(fb):
                # Fallback to bisection
                b = (a + b) / 2.0
                fb = _safe_f(f, b)
                if math.isinf(fb):
                    return b, False
        # Update a, b, c
        if fb * fc < 0:
            a = c
            fa = fc
        else:
            c = a
            fc = fa
        d = b - c
        e = d
        if abs(fb) <= tol:
            return b, True
        if abs(d) < 1e-14:
            return b, True
    
    return b, abs(fb) <= tol


def _find_bracket(f, x0, fx0, deadline):
    """
    Try to find a bracket [a, b] such that f(a) and f(b) have opposite signs.
    Returns (a, b, fa, fb) or None.
    """
    if not math.isfinite(fx0):
        return None
    
    a = x0
    fa = fx0
    
    # Try expanding in both directions
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]:
        for sign in [1.0, -1.0]:
            if time.monotonic() > deadline:
                return None
            b = x0 + sign * scale
            fb = _safe_f(f, b)
            if not math.isfinite(fb):
                continue
            if fa * fb < 0:
                return (min(a, b), max(a, b), fa, fb)
            # If fb is small, we might have a root nearby
            if abs(fb) <= 1e-8:
                return (b, b, fb, fb)
    
    return None


def _newton_polish(f, x, fx, h_base=1e-6, max_iter=20, deadline=None):
    """
    Newton's method polish step for high precision.
    Returns (x, fx, converged).
    """
    for _ in range(max_iter):
        if deadline is not None and time.monotonic() > deadline:
            break
        if abs(fx) <= 1e-14:
            return x, fx, True
        
        h = h_base * max(1.0, abs(x))
        if abs(x) < 1e-8:
            h = 1e-10
        
        fx_plus = _safe_f(f, x + h)
        fx_minus = _safe_f(f, x - h)
        
        if math.isinf(fx_plus) or math.isinf(fx_minus):
            fx_plus = _safe_f(f, x + h)
            if math.isinf(fx_plus):
                break
            d = (fx_plus - fx) / h
        else:
            d = (fx_plus - fx_minus) / (2 * h)
        
        if abs(d) < 1e-16:
            break
        
        step = fx / d
        x_new = x - step
        
        # Limit step size
        max_step = 2.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = math.copysign(max_step, step)
            x_new = x - step
        
        fx_new = _safe_f(f, x_new)
        if math.isinf(fx_new):
            break
        
        x, fx = x_new, fx_new
    
    return x, fx, abs(fx) <= 1e-14


def find_root(f, x0):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    x = float(x0)
    fx = _safe_f(f, x)
    
    if math.isinf(fx):
        # Starting point is bad, try some offsets
        found = False
        for offset in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0]:
            x_try = x + offset
            fx_try = _safe_f(f, x_try)
            if math.isfinite(fx_try):
                if abs(fx_try) <= 1e-8:
                    return x_try
                x, fx = x_try, fx_try
                found = True
                break
        if not found:
            return x
    
    if abs(fx) <= 1e-8:
        return x
    
    # Try to find a bracket and use Brent's method
    bracket = _find_bracket(f, x, fx, deadline)
    
    if bracket is not None:
        a, b, fa, fb = bracket
        if a == b:
            return a
        root, converged = _brent(f, a, b, fa, fb, tol=1e-12, max_iter=100)
        if converged:
            # Polish with Newton for extra precision
            root, fx_root, _ = _newton_polish(f, root, _safe_f(f, root), deadline=deadline)
            if math.isfinite(root) and abs(_safe_f(f, root)) <= 1e-8:
                return root
            return root
    
    # No bracket found; use Newton with damping
    best_x = x
    best_fx = abs(fx)
    
    for iteration in range(50):
        if time.monotonic() > deadline:
            break
        
        if abs(fx) <= 1e-10:
            return x
        
        if abs(fx) < best_fx:
            best_fx = abs(fx)
            best_x = x
        
        h = 1e-6 * max(1.0, abs(x))
        if abs(x) < 1e-8:
            h = 1e-10
        
        fx_plus = _safe_f(f, x + h)
        fx_minus = _safe_f(f, x - h)
        
        if math.isinf(fx_plus) or math.isinf(fx_minus):
            fx_plus = _safe_f(f, x + h)
            if math.isinf(fx_plus):
                break
            d = (fx_plus - fx) / h
        else:
            d = (fx_plus - fx_minus) / (2 * h)
        
        if abs(d) < 1e-16:
            break
        
        step = fx / d
        x_new = x - step
        
        max_step = 5.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = math.copysign(max_step, step)
            x_new = x - step
        
        fx_new = _safe_f(f, x_new)
        
        if math.isinf(fx_new):
            # Try smaller step
            step2 = step * 0.1
            x_new2 = x - step2
            fx_new2 = _safe_f(f, x_new2)
            if math.isfinite(fx_new2) and abs(fx_new2) < abs(fx):
                x, fx = x_new2, fx_new2
            else:
                break
        else:
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                # Damping
                for damp in [0.5, 0.25, 0.1]:
                    x_damped = x - damp * step
                    fx_damped = _safe_f(f, x_damped)
                    if math.isfinite(fx_damped) and abs(fx_damped) < abs(fx):
                        x, fx = x_damped, fx_damped
                        break
                else:
                    break
    
    if abs(fx) <= 1e-8:
        return x
    
    # Final polish
    if abs(fx) < best_fx:
        best_x, best_fx = x, abs(fx)
    
    best_x, fx_best, _ = _newton_polish(f, best_x, _safe_f(f, best_x), deadline=deadline)
    if math.isfinite(best_x) and abs(_safe_f(f, best_x)) <= 1e-8:
        return best_x
    
    if abs(fx) <= 1e-8:
        return x
    return best_x

# EVOLVE-BLOCK-END
