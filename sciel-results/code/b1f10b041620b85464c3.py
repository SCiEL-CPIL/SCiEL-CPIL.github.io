"""Rewritten program with a robust multi-method root finder and time budget guard."""
import math
import time

# Global deadline: evaluator kills after 150s, we use 140s as a safety margin
_DEADLINE = time.time() + 140.0
_CALL_BUDGET = 480  # Stay well under 500 per equation


def _time_left():
    return _DEADLINE - time.time()


def _safe_call(f, x):
    """Call f(x) safely, returning nan on failure."""
    try:
        val = f(x)
        if not math.isfinite(val):
            return float('nan')
        return val
    except Exception:
        return float('nan')


def _newton_fd(f, x, counter, max_iter=30):
    """Newton's method with finite-difference derivative. Returns (x, fx, success)."""
    x = float(x)
    fx = _safe_call(f, x)
    counter[0] += 1
    if math.isnan(fx):
        return x, fx, False
    if abs(fx) <= 1e-8:
        return x, fx, True

    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        h = 1e-7 * max(1.0, abs(x))
        fx2 = _safe_call(f, x + h)
        counter[0] += 1
        if math.isnan(fx2):
            break
        d = (fx2 - fx) / h
        if d == 0 or math.isnan(d):
            break
        step = fx / d
        # Limit step size to avoid jumping too far
        max_step = 1.0 + abs(x)
        if abs(step) > max_step:
            step = max_step * (1 if step > 0 else -1)
        x_new = x - step
        if not math.isfinite(x_new):
            break
        x = x_new
        fx = _safe_call(f, x)
        counter[0] += 1
        if math.isnan(fx):
            break
        if abs(fx) <= 1e-8:
            return x, fx, True
    return x, fx, abs(fx) <= 1e-8


def _bisection(f, a, b, fa, fb, counter, max_iter=100):
    """Bisection method. Returns (x, fx, success)."""
    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if fa * fb > 0:
            break  # No bracket
        c = (a + b) / 2.0
        fc = _safe_call(f, c)
        counter[0] += 1
        if math.isnan(fc):
            break
        if abs(fc) <= 1e-8:
            return c, fc, True
        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        if abs(b - a) < 1e-12 * max(1.0, abs(c)):
            break
    # Return the midpoint of the final bracket
    c = (a + b) / 2.0
    fc = _safe_call(f, c)
    counter[0] += 1
    return c, fc, abs(fc) <= 1e-8


def _brentq_like(f, a, b, fa, fb, counter, max_iter=60):
    """
    Brent's method (simplified) for finding a root in [a, b] where fa*fb < 0.
    Returns (x, fx, success).
    """
    # Ensure fa < 0 and fb > 0 for consistency
    if fa > 0:
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d
    tol = 1e-12

    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if fa * fb > 0:
            break  # Lost bracket
        if fc * fb > 0:
            c = a
            fc = fa
            d = b - a
            e = d
        if abs(e) < tol:
            break  # Converged
        if abs(fc) < abs(fb):
            b = c
            fb = fc
            c = a
            fc = fa
            d = b - a
            e = d

        # Attempt interpolation
        if abs(d) < tol:
            break
        # Secant step
        m = 0.5 * (a - b)
        e2 = 2.0 * m
        p = 2.0 * m * (fc - fb)
        q = (fc - fa)
        if abs(p) >= abs(q):
            # Use secant
            s = p / q
        else:
            # Use quadratic interpolation
            s = m
            q = fc - fa
            r = fc - fb
            if abs(q - r) < 1e-30:
                s = m * (1.0 if p > 0 else -1.0)
            else:
                s = m * (p / (q - r))
        
        # Check if s is acceptable
        if ((3.0 * m - abs(e)) / 2.0 < s < abs(e) / 2.0) or \
           ((e == d and (s < 0 and fc < 0 or s > 0 and fc > 0)) or \
            (abs(s) >= abs(e) / 2.0)):
            s = m
            d = s
        else:
            e = d

        a = b
        fa = fb
        b = b + s
        fb = _safe_call(f, b)
        counter[0] += 1
        if math.isnan(fb):
            break
        if abs(fb) <= 1e-8:
            return b, fb, True

    # Return best estimate
    return b, fb, abs(fb) <= 1e-8


def _find_bracket(f, x0, counter):
    """
    Try to find a bracket [a, b] around x0 where f(a)*f(b) < 0.
    Returns (a, b, fa, fb) or None.
    """
    fx0 = _safe_call(f, x0)
    counter[0] += 1
    if math.isnan(fx0):
        return None
    if abs(fx0) <= 1e-8:
        return x0, x0, fx0, fx0

    # Expand outward to find a sign change
    step = 1.0
    a = x0
    fa = fx0

    for _ in range(30):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        b = x0 + step
        fb = _safe_call(f, b)
        counter[0] += 1
        if math.isnan(fb):
            break
        if fa * fb < 0:
            return a, b, fa, fb
        # Also check the other direction
        b2 = x0 - step
        fb2 = _safe_call(f, b2)
        counter[0] += 1
        if math.isnan(fb2):
            break
        if fa * fb2 < 0:
            return b2, a, fb2, fa
        step *= 2.0

    return None


def find_root(f, x0):
    """
    Find a root x of f(x) = 0 with |f(x)| <= 1e-8.
    Returns x as a float.
    """
    counter = [0]
    x0 = float(x0)
    best_x = x0
    best_fx = _safe_call(f, x0)
    counter[0] += 1
    best_abs = abs(best_fx) if not math.isnan(best_fx) else float('inf')

    if best_abs <= 1e-8:
        return x0

    # Strategy 1: Try Newton's method first (fast if it converges)
    x_newton, fx_newton, success = _newton_fd(f, x0, counter, max_iter=25)
    if not math.isnan(fx_newton):
        if abs(fx_newton) < best_abs:
            best_x = x_newton
            best_abs = abs(fx_newton)
        if success:
            return x_newton

    # Strategy 2: Try to find a bracket and use Brent's method
    bracket = _find_bracket(f, x0, counter)
    if bracket is not None:
        a, b, fa, fb = bracket
        if fa == fb and abs(fa) <= 1e-8:
            return a
        if fa * fb < 0:
            x_brent, fx_brent, success = _brentq_like(f, a, b, fa, fb, counter, max_iter=50)
            if not math.isnan(fx_brent):
                if abs(fx_brent) < best_abs:
                    best_x = x_brent
                    best_abs = abs(fx_brent)
                if success:
                    return x_brent
            # Fallback to bisection if Brent didn't converge
            x_bis, fx_bis, success = _bisection(f, a, b, fa, fb, counter, max_iter=80)
            if not math.isnan(fx_bis):
                if abs(fx_bis) < best_abs:
                    best_x = x_bis
                    best_abs = abs(fx_bis)
                if success:
                    return x_bis

    # Strategy 3: Try more Newton iterations from the best point found
    if best_abs > 1e-8 and not math.isnan(best_fx):
        x_newton2, fx_newton2, success = _newton_fd(f, best_x, counter, max_iter=20)
        if not math.isnan(fx_newton2):
            if abs(fx_newton2) < best_abs:
                best_x = x_newton2
                best_abs = abs(fx_newton2)
            if success:
                return x_newton2

    # Strategy 4: Try a few more bracketing attempts with different starting points
    for offset in [0.1, -0.1, 0.5, -0.5, 2.0, -2.0]:
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        x_try = x0 + offset
        bracket = _find_bracket(f, x_try, counter)
        if bracket is not None:
            a, b, fa, fb = bracket
            if fa * fb < 0:
                x_brent, fx_brent, success = _brentq_like(f, a, b, fa, fb, counter, max_iter=40)
                if not math.isnan(fx_brent):
                    if abs(fx_brent) < best_abs:
                        best_x = x_brent
                        best_abs = abs(fx_brent)
                    if success:
                        return x_brent

    # Return the best result found
    return best_x
