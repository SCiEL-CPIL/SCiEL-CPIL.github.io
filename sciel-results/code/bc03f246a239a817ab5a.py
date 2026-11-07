import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 490
    DEADLINE = time.time() + 140.0

    calls = 0
    best_x = float(x0)
    best_abs = float('inf')

    def safe_f(x):
        nonlocal calls, best_x, best_abs
        if calls >= MAX_CALLS:
            return float('inf')
        if time.time() > DEADLINE:
            return float('inf')
        calls += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            abs_val = abs(val)
            if abs_val < best_abs:
                best_abs = abs_val
                best_x = x
            return val
        except Exception:
            return float('inf')

    x = float(x0)
    fx = safe_f(x)

    if abs(fx) <= TOL:
        return x

    # Strategy: Scan outward with small steps to find a sign change (bracket)
    # For oscillatory functions, small steps are better to catch sign changes
    bracket_a = None
    bracket_b = None
    fa = fx
    fb = None

    # Start with small step, increase gradually
    step = 0.1 * max(1.0, abs(x0))
    for i in range(1, 80):
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        # Try x0 + step
        x_plus = x0 + step
        fp = safe_f(x_plus)
        if math.isfinite(fp) and fa * fp < 0:
            bracket_a = x
            bracket_b = x_plus
            fb = fp
            break
        # Try x0 - step
        x_minus = x0 - step
        fm = safe_f(x_minus)
        if math.isfinite(fm) and fa * fm < 0:
            bracket_a = x_minus
            bracket_b = x
            fb = fm
            break
        # Update fa to the last finite value we evaluated (the one further out)
        if math.isfinite(fp):
            fa = fp
            x = x_plus
        elif math.isfinite(fm):
            fa = fm
            x = x_minus
        step *= 1.5

    if bracket_a is not None:
        # Use brentq for fast convergence
        try:
            result = brentq(safe_f, bracket_a, bracket_b, xtol=1e-15, rtol=1e-15, maxiter=100)
            if math.isfinite(result):
                return result
        except Exception:
            pass

        # Fallback: bisection
        a = bracket_a
        b = bracket_b
        fa_val = safe_f(a)
        fb_val = safe_f(b)

        for _ in range(100):
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            if abs(b - a) < 1e-15:
                break
            c = (a + b) / 2.0
            fc = safe_f(c)
            if not math.isfinite(fc):
                break
            if abs(fc) <= TOL:
                return c
            if fa_val * fc < 0:
                b = c
                fb_val = fc
            else:
                a = c
                fa_val = fc
            if abs(b - a) < 1e-15:
                break

        if abs(fa_val) <= abs(fb_val):
            return a
        else:
            return b
    else:
        # No bracket found. Try Newton's method with damping.
        x = float(x0)
        fx = safe_f(x)
        for _ in range(50):
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            if abs(fx) <= TOL:
                return x
            h = 1e-6 * max(1.0, abs(x))
            d = (safe_f(x + h) - fx) / h
            if not math.isfinite(d) or d == 0:
                break
            dx = fx / d
            # Damp if step is too large
            if abs(dx) > abs(x) * 0.5:
                dx = math.copysign(min(abs(dx), abs(x) * 0.5), dx)
            x_new = x - dx
            if not math.isfinite(x_new):
                break
            x = x_new
            fx = safe_f(x)
        return best_x

    return best_x
