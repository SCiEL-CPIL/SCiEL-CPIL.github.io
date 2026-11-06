"""Root finding: Ill-conditioned - bracket-first hybrid solver using scipy brentq."""
import time
import math
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations of f.
    Strategy:
      1. Fast Newton/secant path if x0 is already close to a root.
      2. Bracket expansion to find a sign change, then scipy brentq.
      3. High-precision polish pass on the best bracket.
      4. Multiple restarts if needed.
    Always returns the best x found (with smallest |f(x)|) within budget.
    """
    start_time = time.time()
    MAX_WALL = 110.0   # Safety margin before 120s limit
    MAX_EVALS = 500
    TOL = 1e-8

    eval_count = [0]
    cache = {}

    def eval_f(x):
        key = round(float(x), 15)
        if key in cache:
            return cache[key]
        if eval_count[0] >= MAX_EVALS:
            return cache.get(key, 0.0)
        eval_count[0] += 1
        try:
            val = float(f(x))
        except Exception:
            val = float('nan')
        cache[key] = val
        return val

    def is_good(fx):
        return abs(fx) <= TOL

    def time_ok():
        return (time.time() - start_time) < MAX_WALL

    def budget_ok():
        return eval_count[0] < MAX_EVALS and time_ok()

    # Track the best point seen so far
    best_x = float(x0)
    best_fx = eval_f(best_x)

    if is_good(best_fx):
        return best_x

    # --- Fast path: damped Newton / secant from x0 if close ---
    def try_newton_secant(x, fx, max_iter=25):
        """Damped Newton with finite-difference derivative, fallback to secant."""
        nonlocal best_x, best_fx
        x = float(x)
        fx = float(fx)
        x_prev = x
        fx_prev = fx
        for i in range(max_iter):
            if not budget_ok():
                break
            if is_good(fx):
                return x, fx
            # Finite-difference derivative
            h = 1e-8 * max(1.0, abs(x))
            fxp = eval_f(x + h)
            if not math.isfinite(fxp):
                break
            d = (fxp - fx) / h
            if not math.isfinite(d) or abs(d) < 1e-300:
                # Try larger h
                h = 1e-6 * max(1.0, abs(x))
                fxp = eval_f(x + h)
                if not math.isfinite(fxp):
                    break
                d = (fxp - fx) / h
                if not math.isfinite(d) or abs(d) < 1e-300:
                    break
            step = -fx / d
            if not math.isfinite(step):
                break
            # Damping: limit step size
            max_step = 2.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = math.copysign(max_step, step)
            x_new = x + step
            fx_new = eval_f(x_new)
            if not math.isfinite(fx_new):
                # Try half step
                x_new = x + 0.5 * step
                fx_new = eval_f(x_new)
                if not math.isfinite(fx_new):
                    break
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
                x_prev, fx_prev = x, fx
            else:
                # No improvement, try secant
                if fx == fx_prev:
                    break
                x_sec = x - fx * (x - x_prev) / (fx - fx_prev)
                if not math.isfinite(x_sec):
                    break
                fx_sec = eval_f(x_sec)
                if not math.isfinite(fx_sec):
                    break
                x, fx = x_sec, fx_sec
                x_prev, fx_prev = x, fx
            if abs(fx) < abs(best_fx):
                best_x, best_fx = x, fx
        return x, fx

    # Fast path
    x_n, fx_n = try_newton_secant(best_x, best_fx, max_iter=20)
    if is_good(fx_n):
        return x_n
    if abs(fx_n) < abs(best_fx):
        best_x, best_fx = x_n, fx_n

    # --- Bracket expansion and brentq ---
    def try_brentq(a, b, fa, fb, max_iter=200):
        """Run scipy brentq with budget guards."""
        nonlocal best_x, best_fx
        if not budget_ok():
            return None, None
        try:
            # Wrap f to count evaluations and guard time
            def f_wrapped(x):
                if not budget_ok():
                    raise RuntimeError("budget")
                return eval_f(x)
            root = brentq(f_wrapped, a, b, xtol=1e-14, rtol=1e-14, maxiter=max_iter)
            fr = eval_f(root)
            if is_good(fr):
                return root, fr
            return root, fr
        except Exception:
            return None, None

    def polish(a, b, fa, fb):
        """High-precision polish: run brentq with tighter tolerance on a small bracket."""
        nonlocal best_x, best_fx
        if not budget_ok():
            return best_x, best_fx
        # Ensure sign change
        if fa * fb > 0:
            return best_x, best_fx
        try:
            def f_wrapped(x):
                if not budget_ok():
                    raise RuntimeError("budget")
                return eval_f(x)
            root = brentq(f_wrapped, a, b, xtol=1e-16, rtol=1e-16, maxiter=100)
            fr = eval_f(root)
            if abs(fr) < abs(best_fx):
                best_x, best_fx = root, fr
        except Exception:
            pass
        return best_x, best_fx

    # Expand bracket around x0
    base_h = 1e-4 * max(1.0, abs(x0))
    found_bracket = False
    for expansion in range(30):
        if not budget_ok():
            break
        h = base_h * (2.0 ** expansion)
        a = x0 - h
        b = x0 + h
        fa = eval_f(a)
        fb = eval_f(b)
        if abs(fa) < abs(best_fx):
            best_x, best_fx = a, fa
        if abs(fb) < abs(best_fx):
            best_x, best_fx = b, fb
        if is_good(fa):
            return a
        if is_good(fb):
            return b
        if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0:
            found_bracket = True
            root, fr = try_brentq(a, b, fa, fb)
            if root is not None and is_good(fr):
                return root
            if root is not None and abs(fr) < abs(best_fx):
                best_x, best_fx = root, fr
            # Polish
            best_x, best_fx = polish(a, b, fa, fb)
            if is_good(best_fx):
                return best_x
            break

    # If no symmetric bracket, try forward/backward secant probes
    if not found_bracket and budget_ok():
        for sign in [1, -1]:
            if not budget_ok():
                break
            h = 1e-3 * max(1.0, abs(x0))
            for expansion in range(20):
                if not budget_ok():
                    break
                x_probe = x0 + sign * h
                fp = eval_f(x_probe)
                if abs(fp) < abs(best_fx):
                    best_x, best_fx = x_probe, fp
                if is_good(fp):
                    return x_probe
                if math.isfinite(best_fx) and math.isfinite(fp) and best_fx * fp < 0:
                    found_bracket = True
                    a, b = (x0, x_probe) if x_probe > x0 else (x_probe, x0)
                    fa, fb = (best_fx, fp) if x_probe > x0 else (fp, best_fx)
                    root, fr = try_brentq(a, b, fa, fb)
                    if root is not None and is_good(fr):
                        return root
                    if root is not None and abs(fr) < abs(best_fx):
                        best_x, best_fx = root, fr
                    best_x, best_fx = polish(a, b, fa, fb)
                    if is_good(best_fx):
                        return best_x
                    break
                h *= 2.0
            if found_bracket:
                break

    # Multiple restarts with different offsets
    if not is_good(best_fx) and budget_ok():
        offsets = [1e-2, -1e-2, 1e-1, -1e-1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0]
        for offset in offsets:
            if not budget_ok() or is_good(best_fx):
                break
            x_try = x0 + offset * max(1.0, abs(x0))
            fx_try = eval_f(x_try)
            if not math.isfinite(fx_try):
                continue
            if abs(fx_try) < abs(best_fx):
                best_x, best_fx = x_try, fx_try
            if is_good(fx_try):
                return x_try
            # Try Newton from here
            x_n2, fx_n2 = try_newton_secant(x_try, fx_try, max_iter=10)
            if is_good(fx_n2):
                return x_n2
            if abs(fx_n2) < abs(best_fx):
                best_x, best_fx = x_n2, fx_n2

    # Final polish if we have a bracket-like pair
    if not is_good(best_fx) and budget_ok():
        # Try a small perturbation to create a bracket for final polish
        h = 1e-10 * max(1.0, abs(best_x))
        for dh in [h, -h, 10*h, -10*h, 100*h, -100*h]:
            if not budget_ok():
                break
            x_p = best_x + dh
            fp = eval_f(x_p)
            if math.isfinite(fp) and math.isfinite(best_fx) and fp * best_fx < 0:
                a, b = (best_x, x_p) if x_p > best_x else (x_p, best_x)
                fa, fb = (best_fx, fp) if x_p > best_x else (fp, best_fx)
                best_x, best_fx = polish(a, b, fa, fb)
                if is_good(best_fx):
                    break

    return best_x

# EVOLVE-BLOCK-END
