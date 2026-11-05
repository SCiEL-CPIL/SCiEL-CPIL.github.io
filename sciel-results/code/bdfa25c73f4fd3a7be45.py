"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Uses:
    1. Bracket search with expanding steps
    2. Brent's method (scipy.optimize.brentq) for bracketed roots
    3. Safeguarded secant / Newton hybrid for unbracketed cases
    4. Multi-start with increasing radii
    5. Deadline guard to stay within 150s limit
    """
    x0 = float(x0)
    deadline = time.monotonic() + 145.0
    max_evals = 495
    eval_count = [0]

    best_x = x0
    best_fx = None
    best_abs = math.inf

    def eval_f(x):
        nonlocal best_x, best_fx, best_abs
        if time.monotonic() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = f(x)
            eval_count[0] += 1
            if not math.isfinite(val):
                return None
            if abs(val) < best_abs:
                best_x = x
                best_fx = val
                best_abs = abs(val)
            return val
        except (ValueError, OverflowError, ZeroDivisionError, TypeError, FloatingPointError):
            eval_count[0] += 1
            return None

    def check_time():
        return time.monotonic() > deadline

    x = x0
    fx = eval_f(x)
    if fx is None:
        return best_x
    if abs(fx) <= 1e-8:
        return x

    # --- Bracket search: expand from x0 to find sign change ---
    def find_bracket(xc, fxc, max_expansions=40):
        """Find [a,b] with f(a)*f(b) < 0 by expanding outward."""
        step = max(1e-3, abs(xc) * 0.1)
        a, b = xc, xc
        fa, fb = fxc, fxc

        for _ in range(max_expansions):
            if check_time() or eval_count[0] >= max_evals:
                return None

            # Expand left
            a_new = a - step
            fa_new = eval_f(a_new)
            if fa_new is not None:
                if fa_new * fb < 0:
                    return (a_new, b, fa_new, fb)
                a, fa = a_new, fa_new

            # Expand right
            b_new = b + step
            fb_new = eval_f(b_new)
            if fb_new is not None:
                if fa * fb_new < 0:
                    return (a, b_new, fa, fb_new)
                b, fb = b_new, fb_new

            step *= 1.7

        return None

    # --- Safeguarded secant/Newton hybrid ---
    def safeguarded_secant(x1, f1, x2, f2, max_iter=30):
        """Secant method with safeguarding via bisection when a bracket is known."""
        # Try to establish a bracket
        a, b = min(x1, x2), max(x1, x2)
        fa, fb = f1 if x1 < x2 else f2, f2 if x1 < x2 else f1

        for i in range(max_iter):
            if check_time() or eval_count[0] >= max_evals:
                break
            if abs(f1) <= 1e-12:
                return x1
            if abs(f2) <= 1e-12:
                return x2

            if f1 == f2:
                break

            # Secant step
            x_new = x2 - f2 * (x2 - x1) / (f2 - f1)

            # Safeguard: if we have a bracket, keep x_new within it
            if fa * fb < 0:
                if x_new < min(a, b) or x_new > max(a, b):
                    x_new = (a + b) / 2

            f_new = eval_f(x_new)
            if f_new is None:
                break

            if abs(f_new) < abs(f1):
                # Update bracket
                if fa * f_new < 0:
                    b, fb = x_new, f_new
                elif fb * f_new < 0:
                    a, fa = x_new, f_new

            x1, f1 = x2, f2
            x2, f2 = x_new, f_new

        best = x1 if abs(f1) <= abs(f2) else x2
        best_f = f1 if abs(f1) <= abs(f2) else f2
        return best if abs(best_f) <= 1e-8 else None

    # --- Newton with finite-difference derivative ---
    def newton_damped(xc, fxc, max_iter=25):
        x = xc
        fx = fxc
        for i in range(max_iter):
            if check_time() or eval_count[0] >= max_evals:
                break
            if abs(fx) <= 1e-12:
                return x

            h = 1e-6 * max(1.0, abs(x))
            fxp = eval_f(x + h)
            if fxp is None:
                break
            d = (fxp - fx) / h
            if d == 0 or not math.isfinite(d):
                break

            step = fx / d
            # Damp if step is large
            if abs(step) > abs(x) * 0.5 + 1e-3:
                step = math.copysign(min(abs(step), abs(x) * 0.5 + 1e-3), step)

            x_new = x - step
            fx_new = eval_f(x_new)
            if fx_new is None:
                break

            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                # Halve step
                step *= 0.5
                x_new = x - step
                fx_new = eval_f(x_new)
                if fx_new is None:
                    break
                if abs(fx_new) < abs(fx):
                    x, fx = x_new, fx_new
                else:
                    break

        return x if abs(fx) <= 1e-8 else None

    # --- Try scipy brentq if bracket found ---
    def try_brentq(a, b, fa, fb):
        try:
            from scipy.optimize import brentq
            # Use our eval_f to count evaluations
            def f_counted(x):
                return eval_f(x)
            root = brentq(f_counted, a, b, maxiter=200, xtol=1e-14, rtol=1e-14)
            if math.isfinite(root):
                return root
        except Exception:
            pass
        return None

    # --- Strategy 1: Find bracket and use brentq ---
    bracket = find_bracket(x, fx)
    if bracket is not None:
        a, b, fa, fb = bracket
        result = try_brentq(a, b, fa, fb)
        if result is not None and abs(eval_f(result) if eval_f(result) is not None else math.inf) <= 1e-8:
            return result
        # If brentq didn't work, fall through

    # --- Strategy 2: Safeguarded secant from x0 ---
    h = 1e-4 * max(1.0, abs(x))
    x2 = x + h
    fx2 = eval_f(x2)
    if fx2 is not None:
        result = safeguarded_secant(x, fx, x2, fx2)
        if result is not None:
            return result

    # --- Strategy 3: Damped Newton from x0 ---
    result = newton_damped(x, fx)
    if result is not None:
        return result

    # --- Strategy 4: Multi-start with increasing radii ---
    rng = np.random.RandomState(12345)
    base_radius = max(1e-3, abs(x0) * 0.01)
    for attempt in range(15):
        if check_time() or eval_count[0] >= max_evals:
            break

        radius = base_radius * (2.0 ** attempt)
        x_try = x0 + rng.normal(0, radius)
        fx_try = eval_f(x_try)
        if fx_try is None:
            continue

        if abs(fx_try) <= 1e-8:
            return x_try

        # Try bracket from this point
        br = find_bracket(x_try, fx_try, max_expansions=15)
        if br is not None:
            a, b, fa, fb = br
            result = try_brentq(a, b, fa, fb)
            if result is not None:
                return result

        # Try secant
        h2 = 1e-5 * max(1.0, abs(x_try))
        x_try2 = x_try + h2
        fx_try2 = eval_f(x_try2)
        if fx_try2 is not None:
            result = safeguarded_secant(x_try, fx_try, x_try2, fx_try2)
            if result is not None:
                return result

        # Try Newton
        result = newton_damped(x_try, fx_try, max_iter=15)
        if result is not None:
            return result

    # --- Strategy 5: Grid search in a local window ---
    if not check_time() and eval_count[0] < max_evals - 20:
        window = max(1e-2, abs(x0) * 0.1)
        n_grid = 20
        for i in range(n_grid):
            if check_time() or eval_count[0] >= max_evals:
                break
            xg = x0 - window + 2 * window * i / (n_grid - 1)
            ffg = eval_f(xg)
            if ffg is None:
                continue
            if abs(ffg) <= 1e-8:
                return xg
            if abs(ffg) < abs(best_fx if best_fx is not None else math.inf):
                best_x, best_fx = xg, ffg
                best_abs = abs(ffg)

    # --- Final polish: if best is close, try to refine ---
    if best_abs <= 1e-5 and best_fx is not None:
        result = newton_damped(best_x, best_fx, max_iter=20)
        if result is not None:
            return result

        h = 1e-7 * max(1.0, abs(best_x))
        xp = best_x + h
        fxp = eval_f(xp)
        if fxp is not None:
            result = safeguarded_secant(best_x, best_fx, xp, fxp, max_iter=20)
            if result is not None:
                return result

    return best_x
# EVOLVE-BLOCK-END
