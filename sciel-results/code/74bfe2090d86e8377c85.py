"""Combined root finder: Secant for fast convergence, scipy brentq for robustness, multi-strategy fallback."""
import math
import time

try:
    from scipy.optimize import brentq as _scipy_brentq
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_DEADLINE = time.time() + 140.0
_CALL_BUDGET = 450


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


def _secant(f, x0, fx0, counter, max_iter=30):
    """Secant method. Returns (x, fx, success)."""
    x1 = x0 + 1e-4 * max(1.0, abs(x0))
    fx1 = _safe_call(f, x1)
    counter[0] += 1
    if math.isnan(fx1):
        return x0, fx0, abs(fx0) <= 1e-8
    if abs(fx1) <= 1e-8:
        return x1, fx1, True

    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if fx0 == fx1:
            break
        step = (x1 - x0) * fx0 / (fx1 - fx0)
        max_step = 1.0 + abs(x0)
        if abs(step) > max_step:
            step = max_step * (1 if step > 0 else -1)
        x_new = x0 - step
        if not math.isfinite(x_new):
            break
        x0, fx0 = x1, fx1
        x1 = x_new
        fx1 = _safe_call(f, x1)
        counter[0] += 1
        if math.isnan(fx1):
            break
        if abs(fx1) <= 1e-8:
            return x1, fx1, True
        if abs(fx0) <= 1e-8:
            return x0, fx0, True
    # Return the better of the two
    if abs(fx0) <= abs(fx1):
        return x0, fx0, abs(fx0) <= 1e-8
    else:
        return x1, fx1, abs(fx1) <= 1e-8


def _bisection(f, a, b, fa, fb, counter, max_iter=80):
    """Bisection method. Returns (x, fx, success)."""
    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if fa * fb > 0:
            break
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
    c = (a + b) / 2.0
    fc = _safe_call(f, c)
    counter[0] += 1
    return c, fc, abs(fc) <= 1e-8


def _brentq_like(f, a, b, fa, fb, counter, max_iter=50):
    """
    Brent's method (simplified) for finding a root in [a, b] where fa*fb < 0.
    Returns (x, fx, success).
    """
    # Ensure |fa| <= |fb|
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d

    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if fa * fb > 0:
            break
        if abs(fa) <= 1e-8:
            return a, fa, True
        if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
            return a, fa, abs(fa) <= 1e-8

        if fc * fb > 0:
            c = a
            fc = fa
            d = b - a
            e = d
        if abs(e) < 1e-12:
            break
        if abs(fc) < abs(fb):
            b = c
            fb = fc
            c = a
            fc = fa
            d = b - a
            e = d

        # Inverse quadratic interpolation
        m = 0.5 * (a - b)
        e2 = 2.0 * m
        p = 2.0 * m * (fc - fb)
        q = (fc - fa)
        if abs(p) >= abs(q):
            s = p / q if q != 0 else m
        else:
            s = m
            q = fc - fa
            r = fc - fb
            if abs(q - r) < 1e-30:
                s = m
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

    # Return best
    if abs(fa) <= abs(fb):
        return a, fa, abs(fa) <= 1e-8
    else:
        return b, fb, abs(fb) <= 1e-8


def _find_bracket(f, x0, fx0, counter):
    """
    Find a bracket [a, b] around x0 where f(a)*f(b) < 0.
    Uses geometric expansion, checking one side at a time to save calls.
    Returns (a, b, fa, fb) or None.
    """
    if math.isnan(fx0):
        return None
    if abs(fx0) <= 1e-8:
        return x0, x0, fx0, fx0

    step = max(1e-3, 0.1 * max(1.0, abs(x0)))
    a = x0
    fa = fx0

    for i in range(25):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        # Alternate right and left to find sign change
        if i % 2 == 0:
            b = x0 + step
            fb = _safe_call(f, b)
            counter[0] += 1
            if not math.isnan(fb):
                if abs(fb) <= 1e-8:
                    return b, b, fb, fb
                if fa * fb < 0:
                    return a, b, fa, fb
        else:
            b = x0 - step
            fb = _safe_call(f, b)
            counter[0] += 1
            if not math.isnan(fb):
                if abs(fb) <= 1e-8:
                    return b, b, fb, fb
                if fa * fb < 0:
                    return b, a, fb, fa
        step *= 2.0
        if step > 1e8:
            break

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

    # Strategy 1: Secant method (fast convergence, 1 call/iteration after setup)
    x_sec, fx_sec, success = _secant(f, x0, best_fx, counter, max_iter=25)
    if not math.isnan(fx_sec):
        if abs(fx_sec) < best_abs:
            best_x = x_sec
            best_abs = abs(fx_sec)
        if success:
            return best_x

    # Strategy 2: Find bracket, then use scipy brentq or our own
    bracket = _find_bracket(f, x0, best_fx, counter)
    if bracket is not None:
        a, b, fa, fb = bracket
        if a == b and abs(fa) <= 1e-8:
            return a
        if fa * fb < 0:
            # Try scipy brentq first (C-level, very efficient)
            if _HAS_SCIPY:
                try:
                    x_br = _scipy_brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                    # Count calls made by scipy (we can't track exactly, so just check result)
                    fx_br = _safe_call(f, x_br)
                    counter[0] += 1
                    if not math.isnan(fx_br):
                        if abs(fx_br) < best_abs:
                            best_x = x_br
                            best_abs = abs(fx_br)
                        if abs(fx_br) <= 1e-8:
                            return best_x
                except Exception:
                    pass
            # Fallback to our own Brent
            x_br, fx_br, success = _brentq_like(f, a, b, fa, fb, counter, max_iter=40)
            if not math.isnan(fx_br):
                if abs(fx_br) < best_abs:
                    best_x = x_br
                    best_abs = abs(fx_br)
                if success:
                    x_pol, fx_pol, _ = _polish(f, x_br, fx_br, counter, max_iter=5)
                    if not math.isnan(fx_pol) and abs(fx_pol) < best_abs:
                        best_x = x_pol
                        best_abs = abs(fx_pol)
                    return best_x
            # Fallback to bisection
            x_bi, fx_bi, success = _bisection(f, a, b, fa, fb, counter, max_iter=60)
            if not math.isnan(fx_bi):
                if abs(fx_bi) < best_abs:
                    best_x = x_bi
                    best_abs = abs(fx_bi)
                if success:
                    x_pol, fx_pol, _ = _polish(f, x_bi, fx_bi, counter, max_iter=5)
                    if not math.isnan(fx_pol) and abs(fx_pol) < best_abs:
                        best_x = x_pol
                        best_abs = abs(fx_pol)
                    return best_x

    # Strategy 3: Try secant from best point
    if best_abs > 1e-8 and not math.isnan(best_fx):
        x_sec2, fx_sec2, success = _secant(f, best_x, best_fx, counter, max_iter=15)
        if not math.isnan(fx_sec2):
            if abs(fx_sec2) < best_abs:
                best_x = x_sec2
                best_abs = abs(fx_sec2)
            if success:
                return best_x

    # Strategy 4: Try more brackets from different starting points
    for offset in [0.1, -0.1, 0.5, -0.5, 2.0, -2.0]:
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        x_try = x0 + offset
        fx_try = _safe_call(f, x_try)
        counter[0] += 1
        if math.isnan(fx_try):
            continue
        if abs(fx_try) < best_abs:
            best_x = x_try
            best_abs = abs(fx_try)
        if abs(fx_try) <= 1e-8:
            return x_try
        bracket = _find_bracket(f, x_try, fx_try, counter)
        if bracket is not None:
            a, b, fa, fb = bracket
            if a == b and abs(fa) <= 1e-8:
                return a
            if fa * fb < 0:
                if _HAS_SCIPY:
                    try:
                        x_br = _scipy_brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                        fx_br = _safe_call(f, x_br)
                        counter[0] += 1
                        if not math.isnan(fx_br):
                            if abs(fx_br) < best_abs:
                                best_x = x_br
                                best_abs = abs(fx_br)
                            if abs(fx_br) <= 1e-8:
                                return best_x
                    except Exception:
                        pass
                x_br, fx_br, success = _brentq_like(f, a, b, fa, fb, counter, max_iter=30)
                if not math.isnan(fx_br):
                    if abs(fx_br) < best_abs:
                        best_x = x_br
                        best_abs = abs(fx_br)
                    if success:
                        return best_x

    return best_x
