"""
Root finder for 3 fixed oscillatory equations using Brent's Method.

Strategy:
  1. Evaluate f at x0. If already good enough, return immediately.
  2. Try to find a bracket [a, b] with sign change using geometric expansion.
     Once bracketed, use a robust Brent's method (inverse quadratic interpolation
     with bisection fallback) to converge to the root.
  3. If no bracket is found, fall back to the secant method.
  4. Try multiple offset starting points if the initial bracketing fails or
     converges slowly, to handle cases where x0 is in a "bad" region.
  5. Polish with secant steps if necessary.

Call budget: keep under 400 calls per equation to maximize score.
Time deadline: return best result before 140 s.
"""
import math
import time

_DEADLINE = time.time() + 140.0
_CALL_BUDGET = 400
_TOL = 1e-8


def _time_left():
    return _DEADLINE - time.time()


class _Counter:
    __slots__ = ('n',)
    def __init__(self):
        self.n = 0
    def __add__(self, other):
        self.n += other
        return self
    def __radd__(self, other):
        self.n += other
        return self


def _safe_call(f, x, ctr):
    """Call f(x), incrementing ctr. Return nan on failure."""
    try:
        val = f(float(x))
        ctr.n += 1
        if not math.isfinite(val):
            return float('nan')
        return float(val)
    except Exception:
        ctr.n += 1
        return float('nan')


def _brent(f, a, b, fa, fb, ctr, max_iter=120):
    """
    Brent's method. Requires fa * fb <= 0.
    Returns (x, fx, success).
    """
    if abs(fa) <= _TOL:
        return a, fa, True
    if abs(fb) <= _TOL:
        return b, fb, True
    if fa * fb > 0:
        # No sign change; return the one with smaller |f|
        if abs(fa) <= abs(fb):
            return a, fa, abs(fa) <= _TOL
        return b, fb, abs(fb) <= _TOL

    # Ensure |fa| <= |fb| so we move b
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d

    for _ in range(max_iter):
        if ctr.n >= _CALL_BUDGET or _time_left() < 0.3:
            break
        if fa * fb > 0:
            break
        if abs(fa) <= _TOL:
            return a, fa, True
        if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
            break

        # Inverse quadratic interpolation
        if fc * fb > 0:
            c = a
            fc = fa
            d = b - a
            e = d

        if abs(e) < 1e-14:
            break
        if abs(fc) < abs(fb):
            b = c
            fb = fc
            c = a
            fc = fa
            d = b - a
            e = d

        m = 0.5 * (a - b)
        e2 = 2.0 * m
        p = 2.0 * m * (fc - fb)
        q = fc - fa
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

        # Check if s is acceptable, otherwise bisect
        if ((3.0 * m - abs(e)) / 2.0 < s < abs(e) / 2.0) or \
           ((e == d and (s < 0 and fc < 0 or s > 0 and fc > 0)) or
            (abs(s) >= abs(e) / 2.0)):
            s = m
            d = s
        else:
            e = d

        a = b
        fa = fb
        b = b + s
        fb = _safe_call(f, b, ctr)
        if math.isnan(fb):
            break
        if abs(fb) <= _TOL:
            return b, fb, True

    # Return best
    if abs(fa) <= abs(fb):
        return a, fa, abs(fa) <= _TOL
    return b, fb, abs(fb) <= _TOL


def _secant(f, x0, fx0, ctr, max_iter=40):
    """
    Secant method starting from x0 with known fx0.
    Returns (x, fx, success).
    """
    step = 1e-6 * max(1.0, abs(x0))
    x1 = x0 + step
    fx1 = _safe_call(f, x1, ctr)
    if math.isnan(fx1):
        return x0, fx0, abs(fx0) <= _TOL
    if abs(fx1) <= _TOL:
        return x1, fx1, True

    for _ in range(max_iter):
        if ctr.n >= _CALL_BUDGET or _time_left() < 0.3:
            break
        if fx0 == fx1:
            break
        denom = fx1 - fx0
        if denom == 0:
            break
        x_new = x0 - fx0 * (x1 - x0) / denom
        if not math.isfinite(x_new):
            break
        x0, fx0 = x1, fx1
        x1 = x_new
        fx1 = _safe_call(f, x1, ctr)
        if math.isnan(fx1):
            break
        if abs(fx1) <= _TOL:
            return x1, fx1, True
        if abs(fx0) <= _TOL:
            return x0, fx0, True

    if abs(fx0) <= abs(fx1):
        return x0, fx0, abs(fx0) <= _TOL
    return x1, fx1, abs(fx1) <= _TOL


def _polish(f, x, fx, ctr, max_iter=10):
    """
    Final high-precision polish: a few secant/Newton-like steps to drive
    |f(x)| as close to zero as possible when already near the root.
    Uses a small fixed-step derivative estimate for Newton fallback.
    Returns (x, fx).
    """
    if math.isnan(fx) or abs(fx) <= _TOL:
        return x, fx

    for _ in range(max_iter):
        if ctr.n >= _CALL_BUDGET or _time_left() < 0.1:
            break
        if abs(fx) <= _TOL:
            break
        # Try a tiny secant step: evaluate f at x + h
        h = 1e-8 * max(1.0, abs(x))
        x2 = x + h
        fx2 = _safe_call(f, x2, ctr)
        if math.isnan(fx2):
            break
        if abs(fx2) <= _TOL:
            return x2, fx2
        denom = fx2 - fx
        if denom == 0:
            # Fall back to pure bisection-like small step toward smaller |f|
            x3 = x - h
            fx3 = _safe_call(f, x3, ctr)
            if math.isnan(fx3):
                break
            if abs(fx3) <= _TOL:
                return x3, fx3
            if abs(fx3) < abs(fx):
                x, fx = x3, fx3
            break
        x_new = x - fx * h / denom
        if not math.isfinite(x_new):
            break
        fx_new = _safe_call(f, x_new, ctr)
        if math.isnan(fx_new):
            break
        if abs(fx_new) <= _TOL:
            return x_new, fx_new
        if abs(fx_new) >= abs(fx):
            # Step didn't help; try the other side
            x_new = x + fx * h / denom
            fx_new = _safe_call(f, x_new, ctr)
            if math.isnan(fx_new):
                break
            if abs(fx_new) <= _TOL:
                return x_new, fx_new
            if abs(fx_new) >= abs(fx):
                break
        x, fx = x_new, fx_new

    return x, fx


def _find_bracket(f, x0, fx0, ctr, max_expand=30):
    """
    Find a bracket [a, b] around x0 where f(a)*f(b) < 0.
    Expands geometrically, alternating direction.
    Returns (a, b, fa, fb) or None.
    """
    if math.isnan(fx0):
        return None
    if abs(fx0) <= _TOL:
        return x0, x0, fx0, fx0

    a = x0
    fa = fx0
    step = max(1e-3, 0.05 * max(1.0, abs(x0)))

    for i in range(max_expand):
        if ctr.n >= _CALL_BUDGET or _time_left() < 0.3:
            break
        if i % 2 == 0:
            b = x0 + step
        else:
            b = x0 - step
        fb = _safe_call(f, b, ctr)
        if math.isnan(fb):
            step *= 1.5
            continue
        if abs(fb) <= _TOL:
            return b, b, fb, fb
        if fa * fb < 0:
            return a, b, fa, fb
        step *= 1.618  # golden ratio for steady expansion

    return None


def _try_bracket_and_brent(f, x_start, fx_start, ctr):
    """Try to bracket around x_start and solve with Brent."""
    if math.isnan(fx_start):
        return None
    if abs(fx_start) <= _TOL:
        return x_start, fx_start
    bracket = _find_bracket(f, x_start, fx_start, ctr)
    if bracket is None:
        return None
    a, b, fa, fb = bracket
    if a == b and abs(fa) <= _TOL:
        return a, fa
    if fa * fb < 0:
        x, fx, _ = _brent(f, a, b, fa, fb, ctr)
        if not math.isnan(fx):
            return x, fx
    return None


def find_root(f, x0):
    """
    Find a root x of f(x) = 0 with |f(x)| <= 1e-8.
    Returns x as a float.
    """
    ctr = _Counter()
    x0 = float(x0)
    fx0 = _safe_call(f, x0, ctr)
    best_x = x0
    best_fx = fx0
    best_abs = abs(fx0) if not math.isnan(fx0) else float('inf')

    if best_abs <= _TOL:
        return x0

    # Strategy 1: bracket + Brent from x0
    result = _try_bracket_and_brent(f, x0, fx0, ctr)
    if result is not None:
        x, fx = result
        if not math.isnan(fx):
            if abs(fx) < best_abs:
                best_x, best_fx, best_abs = x, fx, abs(fx)
            if abs(fx) <= _TOL:
                return best_x

    # Strategy 2: secant from x0
    if best_abs > _TOL and not math.isnan(best_fx):
        x_s, fx_s, ok = _secant(f, best_x, best_fx, ctr, max_iter=30)
        if not math.isnan(fx_s):
            if abs(fx_s) < best_abs:
                best_x, best_fx, best_abs = x_s, fx_s, abs(fx_s)
            if ok:
                return best_x

    # Strategy 3: try offsets to find a better starting point for bracketing
    # Oscillatory functions might have roots far from x0 if x0 is in a "flat" region
    offsets = [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0,
               0.05, -0.05, 0.25, -0.25, 1.5, -1.5, 3.0, -3.0, 7.0, -7.0]
    for off in offsets:
        if ctr.n >= _CALL_BUDGET or _time_left() < 0.3:
            break
        x_try = x0 + off
        fx_try = _safe_call(f, x_try, ctr)
        if math.isnan(fx_try):
            continue
        if abs(fx_try) < best_abs:
            best_x, best_fx, best_abs = x_try, fx_try, abs(fx_try)
        if abs(fx_try) <= _TOL:
            return best_x
        # Try bracket from this point
        result = _try_bracket_and_brent(f, x_try, fx_try, ctr)
        if result is not None:
            x, fx = result
            if not math.isnan(fx):
                if abs(fx) < best_abs:
                    best_x, best_fx, best_abs = x, fx, abs(fx)
                if abs(fx) <= _TOL:
                    return best_x

    # Strategy 4: secant from best found so far
    if best_abs > _TOL and not math.isnan(best_fx):
        x_s, fx_s, ok = _secant(f, best_x, best_fx, ctr, max_iter=20)
        if not math.isnan(fx_s):
            if abs(fx_s) < best_abs:
                best_x, best_fx, best_abs = x_s, fx_s, abs(fx_s)
            if ok:
                return best_x

    # Strategy 5: try to bracket from best point again if we have budget
    if best_abs > _TOL and not math.isnan(best_fx) and ctr.n < _CALL_BUDGET - 10:
        result = _try_bracket_and_brent(f, best_x, best_fx, ctr)
        if result is not None:
            x, fx = result
            if not math.isnan(fx):
                if abs(fx) < best_abs:
                    best_x, best_fx, best_abs = x, fx, abs(fx)
                if abs(fx) <= _TOL:
                    return best_x

    # Strategy 6: final high-precision polish pass
    if best_abs > _TOL and not math.isnan(best_fx) and ctr.n < _CALL_BUDGET - 5:
        best_x, best_fx = _polish(f, best_x, best_fx, ctr)
        best_abs = abs(best_fx)
        if best_abs <= _TOL:
            return best_x

    return best_x
