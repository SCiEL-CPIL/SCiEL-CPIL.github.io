"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import brentq, newton

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 145.0  # safety margin before 150s kill

    evals = 0
    best_x = float(x0)
    best_abs = float('inf')

    def call_f(x):
        nonlocal evals, best_x, best_abs
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return None
        if math.isfinite(val) and abs(val) < best_abs:
            best_abs = abs(val)
            best_x = x
        return val

    def is_good(x, fx):
        return fx is not None and math.isfinite(fx) and abs(fx) <= TOL

    # --- Hand-rolled fallback solvers (kept from current program) ---
    def newton_raphson(x0, max_iters=40):
        x = float(x0)
        for _ in range(max_iters):
            if time.monotonic() > DEADLINE:
                return None
            fx = call_f(x)
            if fx is None:
                return None
            if abs(fx) <= TOL:
                return x
            h = 1e-7 * max(1.0, abs(x))
            fx2 = call_f(x + h)
            if fx2 is None:
                return None
            d = (fx2 - fx) / h
            if d == 0 or not math.isfinite(d):
                return None
            x_new = x - fx / d
            if not math.isfinite(x_new):
                return None
            # Damping if step is too large
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x + 10 * max(1.0, abs(x)) * (1 if x_new > x else -1)
            x = x_new
        return None

    def secant(x0, max_iters=40):
        x0 = float(x0)
        h = 1e-4 * max(1.0, abs(x0))
        x1 = x0 + h
        f0 = call_f(x0)
        f1 = call_f(x1)
        if f0 is None or f1 is None:
            return None
        if abs(f0) <= TOL:
            return x0
        if abs(f1) <= TOL:
            return x1
        for _ in range(max_iters):
            if time.monotonic() > DEADLINE:
                return None
            if f1 == f0:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            if not math.isfinite(x2):
                break
            f2 = call_f(x2)
            if f2 is None:
                break
            if abs(f2) <= TOL:
                return x2
            x0, x1 = x1, x2
            f0, f1 = f1, f2
        return None

    def find_bracket(x, fx):
        if fx is None:
            return None
        if abs(fx) <= TOL:
            return (x, x, fx, fx)
        step = 0.1 * max(1.0, abs(x))
        for direction in (1.0, -1.0):
            b = x + direction * step
            fb = call_f(b)
            if fb is None:
                continue
            if fx * fb < 0.0:
                if direction > 0:
                    return (x, b, fx, fb)
                else:
                    return (b, x, fb, fx)
            cur = b
            curf = fb
            for _ in range(60):
                if time.monotonic() > DEADLINE:
                    return None
                step *= 2.0
                nxt = x + direction * step
                fn = call_f(nxt)
                if fn is None:
                    break
                if curf * fn < 0.0:
                    if direction > 0:
                        return (cur, nxt, curf, fn)
                    else:
                        return (nxt, cur, fn, curf)
                cur, curf = nxt, fn
        return None

    def illinois(lo, hi, flo, fhi):
        for _ in range(80):
            if time.monotonic() > DEADLINE:
                return None
            if flo * fhi > 0.0:
                break
            if abs(flo) < abs(fhi):
                lo, hi = hi, lo
                flo, fhi = fhi, flo
            denom = fhi - flo
            if abs(denom) < 1e-300:
                mid = 0.5 * (lo + hi)
            else:
                mid = (flo * hi - fhi * lo) / denom
                if not (lo < mid < hi):
                    mid = 0.5 * (lo + hi)
            fm = call_f(mid)
            if fm is None:
                break
            if abs(fm) <= TOL:
                return mid
            if flo * fm < 0.0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, 0.5 * flo
            if abs(hi - lo) <= 1e-14 * max(1.0, abs(mid)):
                return mid
        return None

    def muller(x0):
        x = float(x0)
        fx = call_f(x)
        if fx is None:
            return None
        h = 1e-4 * max(1.0, abs(x))
        x1 = x + h
        f1 = call_f(x1)
        if f1 is None:
            return None
        x2 = x1 + h
        f2 = call_f(x2)
        if f2 is None:
            return None
        if abs(fx) <= TOL:
            return x
        if abs(f1) <= TOL:
            return x1
        if abs(f2) <= TOL:
            return x2
        for _ in range(80):
            if time.monotonic() > DEADLINE:
                return None
            d0 = f1 - f2
            d1 = fx - f2
            c = f2
            b = d0
            a = d1
            if a == 0:
                break
            disc = b * b - 4.0 * a * c
            if disc < 0:
                break
            sq = math.sqrt(disc)
            r1 = (-b + sq) / (2.0 * a)
            r2 = (-b - sq) / (2.0 * a)
            if abs(r1) > abs(r2):
                s = r1
            else:
                s = r2
            if s == 0:
                break
            x_new = x2 + s
            if not math.isfinite(x_new):
                break
            f_new = call_f(x_new)
            if f_new is None:
                break
            if abs(f_new) <= TOL:
                return x_new
            x, fx = x2, f2
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
        return None

    # --- SciPy-based solvers (new) ---
    def scipy_brentq(lo, hi):
        if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 1:
            return None
        try:
            # brentq uses our call_f, so evals are counted
            root = brentq(call_f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=100)
            if math.isfinite(root):
                return root
        except Exception:
            pass
        return None

    def scipy_newton(x0):
        if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 1:
            return None
        try:
            # Use finite-difference derivative via call_f
            def df(x):
                h = 1e-6 * max(1.0, abs(x))
                f1 = call_f(x + h)
                f2 = call_f(x - h)
                if f1 is None or f2 is None:
                    return 1.0  # fallback
                return (f1 - f2) / (2 * h)
            root = newton(call_f, x0, fprime=df, tol=1e-12, maxiter=50)
            if math.isfinite(root):
                return root
        except Exception:
            pass
        return None

    # --- Main strategy ---
    x0 = float(x0)

    # 1) Try SciPy Newton first (fast if close to root)
    result = scipy_newton(x0)
    if result is not None and is_good(result, call_f(result)):
        return result

    # 2) Try hand-rolled Newton (fallback)
    result = newton_raphson(x0)
    if result is not None:
        return result

    # 3) Try secant
    result = secant(x0)
    if result is not None:
        return result

    # 4) Try Muller
    result = muller(x0)
    if result is not None:
        return result

    # 5) Try bracketing + SciPy Brent
    fx0 = call_f(x0)
    bracket = find_bracket(x0, fx0)
    if bracket is not None:
        lo, hi, flo, fhi = bracket
        if lo != hi:  # non-degenerate bracket
            result = scipy_brentq(lo, hi)
            if result is not None:
                return result
        # Fallback to Illinois
        result = illinois(lo, hi, flo, fhi)
        if result is not None:
            return result

    # 6) Try Newton with different initial points
    for offset in [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
        if time.monotonic() > DEADLINE:
            break
        x_try = x0 + offset
        result = scipy_newton(x_try)
        if result is not None and is_good(result, call_f(result)):
            return result
        result = newton_raphson(x_try, max_iters=20)
        if result is not None:
            return result
        result = secant(x_try, max_iters=20)
        if result is not None:
            return result

    # Return best found
    return best_x
# EVOLVE-BLOCK-END
