"""Root finding for ill-conditioned equations: robust adaptive hybrid."""
import time
import math
import numpy as np
from scipy.optimize import brentq

# Global deadline: leave a safety margin before the 150s kill.
_DEADLINE_SECONDS = 135.0

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8, using at most 500 evaluations of f.
    Returns the best x found (smallest |f(x)|) within the budget.
    """
    start_time = time.time()
    eval_count = 0
    max_evals = 500
    tol = 1e-8

    # State to track the best point seen so far
    best_x = float(x0)
    best_fx = f(best_x)
    eval_count += 1
    best_abs_fx = abs(best_fx)

    def _time_left():
        return _DEADLINE_SECONDS - (time.time() - start_time)

    def _eval(x):
        """Evaluate f(x) with eval-count and time guards. Returns (fx, ok)."""
        nonlocal eval_count, best_x, best_fx, best_abs_fx
        if eval_count >= max_evals or _time_left() <= 0.5:
            return None, False
        try:
            fx = float(f(x))
        except Exception:
            return None, False
        eval_count += 1
        if not math.isfinite(fx):
            return None, False
        # Update best
        if abs(fx) < best_abs_fx:
            best_x = x
            best_fx = fx
            best_abs_fx = abs(fx)
        return fx, True

    def _check_done():
        return best_abs_fx <= tol

    # If already good enough
    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 1: Inverse quadratic interpolation (history-based)
    # Builds a quadratic model of f(x) from up to 3 recent points and
    # steps to a predicted root. Falls back to secant if only 2 points.
    # This reuses all previous function values, requiring only one
    # evaluation per iteration to update the inverse model.
    # ------------------------------------------------------------------
    x = float(x0)
    fx, _ = _eval(x)
    if fx is None:
        return best_x

    # History: list of (x, fx) tuples, most recent last
    history = [(x, fx)]

    # Helper: compute next point from history via inverse interpolation
    def _interp_next(hist):
        """
        Given a list of (x, fx) pairs (len 2 or 3), compute the next
        predicted x via inverse quadratic interpolation (or secant if len 2).
        Returns the new x, or None if not possible.
        """
        n = len(hist)
        if n < 2:
            return None

        # Sort by x for numerical stability
        pts = sorted(hist, key=lambda p: p[0])

        if n == 2:
            x1, f1 = pts[0]
            x2, f2 = pts[1]
            if abs(x2 - x1) < 1e-15:
                return None
            d = (f2 - f1) / (x2 - x1)
            if abs(d) < 1e-15:
                return None
            return x2 - f2 / d

        # n == 3: inverse quadratic interpolation
        x1, f1 = pts[0]
        x2, f2 = pts[1]
        x3, f3 = pts[2]

        # Check for near-duplicate x values
        if abs(x2 - x1) < 1e-14 or abs(x3 - x2) < 1e-14 or abs(x3 - x1) < 1e-14:
            return None

        # Lagrange interpolation: find x where p(x) = 0
        d12 = x1 - x2
        d13 = x1 - x3
        d23 = x2 - x3

        if abs(d12) < 1e-14 or abs(d13) < 1e-14 or abs(d23) < 1e-14:
            return None

        c1 = f1 / (d12 * d13)
        c2 = f2 / ((x2 - x1) * (x2 - x3))
        c3 = f3 / ((x3 - x1) * (x3 - x2))

        # Quadratic coefficients: A*x^2 + B*x + C = 0
        A = c1 + c2 + c3
        B = -c1 * (x2 + x3) - c2 * (x1 + x3) - c3 * (x1 + x2)
        C = c1 * x2 * x3 + c2 * x1 * x3 + c3 * x1 * x2

        if abs(A) < 1e-30:
            # Degenerate to linear
            if abs(B) < 1e-30:
                return None
            return -C / B

        disc = B * B - 4.0 * A * C
        if disc < 0:
            return None

        sqrt_disc = math.sqrt(disc)
        x_root1 = (-B + sqrt_disc) / (2.0 * A)
        x_root2 = (-B - sqrt_disc) / (2.0 * A)

        # Choose the root closer to the most recent point for stability
        x_recent = hist[-1][0]
        if abs(x_root1 - x_recent) <= abs(x_root2 - x_recent):
            return x_root1
        else:
            return x_root2

    # Main inverse-interpolation loop
    max_interp_iters = 60
    interp_iters = 0
    while interp_iters < max_interp_iters and eval_count < max_evals and _time_left() > 1.0:
        if _check_done():
            return best_x

        x_new = _interp_next(history)
        if x_new is None:
            break

        # Safeguard: limit step to avoid runaway
        max_step = 20.0 * max(1.0, abs(x))
        if abs(x_new - x) > max_step:
            x_new = x + math.copysign(max_step, x_new - x)

        fx_new, ok = _eval(x_new)
        if not ok:
            break

        if abs(fx_new) <= tol:
            return x_new

        # Update history: keep last 3 points
        history.append((x_new, fx_new))
        if len(history) > 3:
            history.pop(0)

        # Check for stagnation
        if len(history) >= 2:
            if abs(history[-1][0] - history[-2][0]) < 1e-12 and abs(history[-1][1] - history[-2][1]) < 1e-12:
                break

        x, fx = x_new, fx_new
        interp_iters += 1

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 2: Safeguarded Newton / Secant hybrid
    # ------------------------------------------------------------------
    x = float(x0)
    fx, _ = _eval(x)
    if fx is None:
        return best_x

    # Finite difference derivative
    def _fd_deriv(x, fx):
        h = 1e-7 * max(1.0, abs(x))
        fxp, ok = _eval(x + h)
        if not ok:
            return None
        return (fxp - fx) / h

    newton_iters = 0
    max_newton = 30
    prev_x = x
    prev_fx = fx

    while newton_iters < max_newton and eval_count < max_evals and _time_left() > 1.0:
        if _check_done():
            return best_x

        # Try Newton step
        d = _fd_deriv(x, fx)
        if d is None or abs(d) < 1e-15:
            # Fallback to secant if we have a previous point
            if newton_iters > 0 and abs(x - prev_x) > 1e-15:
                d = (fx - prev_fx) / (x - prev_x)
            if d is None or abs(d) < 1e-15:
                break

        step = fx / d
        x_new = x - step

        # Safeguard: limit step size to avoid runaway
        max_step = 10.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            x_new = x - math.copysign(max_step, step)

        fx_new, ok = _eval(x_new)
        if not ok:
            break

        # Check for convergence
        if abs(fx_new) <= tol:
            return x_new

        # Check for progress
        if abs(fx_new) >= abs(fx) * 0.9 and abs(x_new - x) < 1e-10:
            # Stagnating, break to try bracketing
            break

        # Accept the step
        prev_x, prev_fx = x, fx
        x, fx = x_new, fx_new
        newton_iters += 1

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 3: Bracketing + Brent's method
    # ------------------------------------------------------------------
    # Try to find a bracket [a, b] with sign change around current best
    a = x
    fa = fx
    if fa is None:
        fa, _ = _eval(a)
        if fa is None:
            return best_x

    # Expand bracket outward
    b = None
    fb = None
    scale = 1.0
    for _ in range(20):
        if _time_left() < 1.0 or eval_count >= max_evals - 10:
            break
        b = a + scale * max(1.0, abs(a))
        fb, ok = _eval(b)
        if not ok:
            break
        if fa * fb < 0:
            break
        # Also try the other direction
        b_neg = a - scale * max(1.0, abs(a))
        fb_neg, ok2 = _eval(b_neg)
        if ok2 and fa * fb_neg < 0:
            b = b_neg
            fb = fb_neg
            break
        scale *= 2.0

    if b is not None and fb is not None and fa * fb < 0:
        # Use Brent's method
        try:
            # brentq uses f directly; we need to wrap to count evals
            def wrapped_f(x_val):
                nonlocal eval_count
                if eval_count >= max_evals or _time_left() < 0.5:
                    raise RuntimeError("budget exceeded")
                fx_val = float(f(x_val))
                eval_count += 1
                if not math.isfinite(fx_val):
                    raise RuntimeError("non-finite")
                if abs(fx_val) < best_abs_fx:
                    best_x = x_val
                    best_fx = fx_val
                    best_abs_fx = abs(fx_val)
                return fx_val

            x_root = brentq(wrapped_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if abs(wrapped_f(x_root)) <= tol:
                return x_root
        except Exception:
            pass

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 4: Secant method from current point
    # ------------------------------------------------------------------
    # Use two recent points for secant
    x1 = x
    f1 = fx
    x2 = x + 1e-6 * max(1.0, abs(x))
    f2, ok = _eval(x2)
    if not ok:
        return best_x

    secant_iters = 0
    max_secant = 50
    while secant_iters < max_secant and eval_count < max_evals and _time_left() > 1.0:
        if _check_done():
            return best_x
        if abs(x1 - x2) < 1e-15:
            break
        d = (f1 - f2) / (x1 - x2)
        if abs(d) < 1e-15:
            break
        x_new = x2 - f2 / d
        f_new, ok = _eval(x_new)
        if not ok:
            break
        if abs(f_new) <= tol:
            return x_new
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new
        secant_iters += 1

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 5: Small restart portfolio
    # ------------------------------------------------------------------
    # Try a few deterministic offsets from x0
    offsets = [1e-4, -1e-4, 1e-3, -1e-3, 0.1, -0.1, 1.0, -1.0]
    for off in offsets:
        if eval_count >= max_evals - 20 or _time_left() < 2.0:
            break
        x_try = float(x0) + off * max(1.0, abs(x0))
        fx_try, ok = _eval(x_try)
        if not ok:
            continue
        if abs(fx_try) <= tol:
            return x_try

        # Quick Newton from this point
        for _ in range(10):
            if _check_done():
                return best_x
            if eval_count >= max_evals or _time_left() < 1.0:
                break
            d = _fd_deriv(x_try, fx_try)
            if d is None or abs(d) < 1e-15:
                break
            step = fx_try / d
            x_new = x_try - step
            max_step = 10.0 * max(1.0, abs(x_try))
            if abs(step) > max_step:
                x_new = x_try - math.copysign(max_step, step)
            fx_new, ok = _eval(x_new)
            if not ok:
                break
            if abs(fx_new) <= tol:
                return x_new
            x_try, fx_try = x_new, fx_new

    return best_x
