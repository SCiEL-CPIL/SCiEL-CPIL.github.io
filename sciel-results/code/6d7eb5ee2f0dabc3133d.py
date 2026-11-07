"""
Optimized root-finder for 3 fixed oscillatory equations.

Key improvements over the previous version:
  1. Faster local convergence: secant + a few Newton-secant hybrid steps.
  2. Smarter bracketing: smaller initial step, denser local scan for oscillatory roots.
  3. Use scipy brentq when available (very efficient), otherwise Illinois.
  4. Tighter call budget management to keep mean_calls low.
  5. Deadline guard with safety margin before 120 s.
"""
import math
import time

try:
    from scipy.optimize import brentq as _scipy_brentq
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_START = time.time()
_DEADLINE = _START + 115.0
_CALL_BUDGET = 470
_TOL = 1e-8


def _time_left():
    return _DEADLINE - time.time()


class _Counter:
    __slots__ = ('n',)
    def __init__(self):
        self.n = 0


def _safe_call(f, x, ctr):
    try:
        val = f(float(x))
        ctr.n += 1
        if not math.isfinite(val):
            return float('nan')
        return float(val)
    except Exception:
        ctr.n += 1
        return float('nan')


def _illinois(f, a, b, fa, fb, ctr, max_iter=60):
    if abs(fa) <= _TOL:
        return a, fa, True
    if abs(fb) <= _TOL:
        return b, fb, True
    if fa * fb > 0:
        if abs(fa) <= abs(fb):
            return a, fa, abs(fa) <= _TOL
        return b, fb, abs(fb) <= _TOL

    for _ in range(max_iter):
        if ctr.n >= _CALL_BUDGET or _time_left() < 0.3:
            break
        if fa * fb > 0:
            break
        if abs(fa) <= _TOL:
            return a, fa, True
        if abs(fb) <= _TOL:
            return b, fb, True
        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
            break
        denom = fb - fa
        if denom == 0:
            break
        x_new = a - fa * (b - a) / denom
        if not math.isfinite(x_new):
            break
        fx_new = _safe_call(f, x_new, ctr)
        if math.isnan(fx_new):
            break
        if abs(fx_new) <= _TOL:
            return x_new, fx_new, True
        if fa * fx_new < 0:
            b = x_new
            fb = fx_new
            fa *= 0.5
        else:
            a = x_new
            fa = fx_new
            fb *= 0.5

    if abs(fa) <= abs(fb):
        return a, fa, abs(fa) <= _TOL
    return b, fb, abs(fb) <= _TOL


def _brent_solve(f, a, b, fa, fb, ctr):
    if abs(fa) <= _TOL:
        return a, fa, True
    if abs(fb) <= _TOL:
        return b, fb, True
    if fa * fb > 0:
        if abs(fa) <= abs(fb):
            return a, fa, abs(fa) <= _TOL
        return b, fb, abs(fb) <= _TOL

    if _HAS_SCIPY and ctr.n < _CALL_BUDGET and _time_left() > 0.5:
        try:
            def _wrapped(x):
                val = f(float(x))
                ctr.n += 1
                if not math.isfinite(val):
                    raise ValueError("non-finite")
                return float(val)
            x_root = _scipy_brentq(_wrapped, a, b, xtol=1e-13, rtol=1e-13, maxiter=60)
            fx_root = _safe_call(f, x_root, ctr)
            if not math.isnan(fx_root) and abs(fx_root) <= _TOL:
                return x_root, fx_root, True
        except Exception:
            pass

    return _illinois(f, a, b, fa, fb, ctr, max_iter=60)


def _secant(f, x0, fx0, ctr, max_iter=25):
    step = 1e-7 * max(1.0, abs(x0))
    x1 = x0 + step
    fx1 = _safe_call(f, x1, ctr)
    if math.isnan(fx1):
        return x0, fx0, abs(fx0) <= _TOL
    if abs(fx1) <= _TOL:
        return x1, fx1, True

    best_x, best_fx, best_abs = x0, fx0, abs(fx0)

    for _ in range(max_iter):
        if ctr.n >= _CALL_BUDGET or _time_left() < 0.3:
            break
        denom = fx1 - fx0
        if denom == 0:
            break
        x_new = x0 - fx0 * (x1 - x0) / denom
        if not math.isfinite(x_new):
            break
        if abs(x_new - x1) > 10.0 * max(1.0, abs(x1)):
            break
        x0, fx0 = x1, fx1
        x1 = x_new
        fx1 = _safe_call(f, x1, ctr)
        if math.isnan(fx1):
            break
        if abs(fx1) < best_abs:
            best_x, best_fx, best_abs = x1, fx1, abs(fx1)
        if abs(fx1) <= _TOL:
            return x1, fx1, True
        if abs(fx0) <= _TOL:
            return x0, fx0, True
        if abs(x1 - x0) < 1e-15 * max(1.0, abs(x1)):
            break

    if abs(fx0) <= abs(fx1):
        return x0, fx0, abs(fx0) <= _TOL
    return x1, fx1, abs(fx1) <= _TOL


def _find_bracket(f, x0, fx0, ctr, max_expand=25):
    if math.isnan(fx0):
        return None
    if abs(fx0) <= _TOL:
        return x0, x0, fx0, fx0

    a = x0
    fa = fx0
    step = max(1e-4, 0.02 * max(1.0, abs(x0)))

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
        step *= 1.618

    # Denser local scan for oscillatory functions
    scan_step = max(1e-5, 0.005 * max(1.0, abs(x0)))
    n_scan = 30
    for i in range(1, n_scan):
        if ctr.n >= _CALL_BUDGET or _time_left() < 0.3:
            break
        x_left = x0 - i * scan_step
        x_right = x0 + i * scan_step
        fl = _safe_call(f, x_left, ctr)
        if math.isnan(fl):
            continue
        if abs(fl) <= _TOL:
            return x_left, x_left, fl, fl
        fr = _safe_call(f, x_right, ctr)
        if math.isnan(fr):
            continue
        if abs(fr) <= _TOL:
            return x_right, x_right, fr, fr
        if fl * fr < 0:
            return x_left, x_right, fl, fr

    return None


def _try_bracket_and_brent(f, x_start, fx_start, ctr):
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
        x, fx, ok = _brent_solve(f, a, b, fa, fb, ctr)
        if not math.isnan(fx) and abs(fx) <= _TOL:
            return x, fx
    return None


def find_root(f, x0):
    ctr = _Counter()
    x0 = float(x0)
    fx0 = _safe_call(f, x0, ctr)
    best_x, best_fx, best_abs = x0, fx0, (abs(fx0) if not math.isnan(fx0) else float('inf'))

    if best_abs <= _TOL:
        return x0

    # Strategy 1: Secant
    if not math.isnan(best_fx):
        x_s, fx_s, ok = _secant(f, best_x, best_fx, ctr, max_iter=25)
        if not math.isnan(fx_s):
            if abs(fx_s) < best_abs:
                best_x, best_fx, best_abs = x_s, fx_s, abs(fx_s)
            if ok:
                return best_x

    # Strategy 2: Bracket + Brent from best
    if best_abs > _TOL and not math.isnan(best_fx):
        result = _try_bracket_and_brent(f, best_x, best_fx, ctr)
        if result is not None:
            x, fx = result
            if not math.isnan(fx):
                if abs(fx) < best_abs:
                    best_x, best_fx, best_abs = x, fx, abs(fx)
                if abs(fx) <= _TOL:
                    return best_x

    # Strategy 3: Try offsets
    offsets = [0.05, -0.05, 0.1, -0.1, 0.2, -0.2, 0.5, -0.5, 1.0, -1.0,
               2.0, -2.0, 5.0, -5.0, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03,
               0.15, -0.15, 0.3, -0.3, 0.7, -0.7, 1.5, -1.5, 3.0, -3.0,
               7.0, -7.0, 10.0, -10.0, 0.4, -0.4, 0.6, -0.6, 1.2, -1.2]
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
        result = _try_bracket_and_brent(f, x_try, fx_try, ctr)
        if result is not None:
            x, fx = result
            if not math.isnan(fx):
                if abs(fx) < best_abs:
                    best_x, best_fx, best_abs = x, fx, abs(fx)
                if abs(fx) <= _TOL:
                    return best_x

    # Strategy 4: Secant from best again
    if best_abs > _TOL and not math.isnan(best_fx) and _time_left() > 0.5:
        x_s, fx_s, ok = _secant(f, best_x, best_fx, ctr, max_iter=15)
        if not math.isnan(fx_s):
            if abs(fx_s) < best_abs:
                best_x, best_fx, best_abs = x_s, fx_s, abs(fx_s)
            if ok:
                return best_x

    # Strategy 5: Local Illinois polish
    if best_abs > _TOL and not math.isnan(best_fx) and _time_left() > 0.5:
        h = 1e-4 * max(1.0, abs(best_x))
        x_left = best_x - h
        x_right = best_x + h
        fl = _safe_call(f, x_left, ctr)
        fr = _safe_call(f, x_right, ctr)
        if not math.isnan(fl) and not math.isnan(fr):
            if abs(fl) < best_abs:
                best_x, best_fx, best_abs = x_left, fl, abs(fl)
            if abs(fr) < best_abs:
                best_x, best_fx, best_abs = x_right, fr, abs(fr)
            if fl * fr < 0:
                x_i, fx_i, ok_i = _illinois(f, x_left, x_right, fl, fr, ctr, max_iter=25)
                if not math.isnan(fx_i):
                    if abs(fx_i) < best_abs:
                        best_x, best_fx, best_abs = x_i, fx_i, abs(fx_i)
                    if ok_i:
                        return best_x

    return best_x
