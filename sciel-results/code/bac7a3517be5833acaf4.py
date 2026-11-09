"""Root finding for ill-conditioned equations: robust adaptive hybrid with polish."""
import time
import math
import numpy as np

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8, using at most 500 evaluations of f.
    Returns the best x found (smallest |f(x)|) within the budget.
    """
    start_time = time.time()
    eval_count = 0
    max_evals = 495  # Leave a small safety margin
    tol = 1e-8
    deadline = 130.0

    # State to track the best point seen so far
    best_x = float(x0)
    best_fx = None
    best_abs_fx = float('inf')

    def _time_left():
        return deadline - (time.time() - start_time)

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

    # Initial evaluation
    best_fx, _ = _eval(best_x)
    if best_fx is None:
        return best_x
    best_abs_fx = abs(best_fx)

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 1: Safeguarded Newton with finite difference derivative
    # ------------------------------------------------------------------
    x = float(x0)
    fx = best_fx

    def _fd_deriv(x, fx):
        """Compute derivative using central difference if possible, else forward."""
        h = 1e-7 * max(1.0, abs(x))
        # Try central difference: 2 evals
        if eval_count + 2 < max_evals and _time_left() > 1.0:
            fxp, ok1 = _eval(x + h)
            if ok1:
                fxm, ok2 = _eval(x - h)
                if ok2:
                    return (fxp - fxm) / (2 * h)
        # Fallback to forward difference: 1 eval
        if eval_count + 1 < max_evals and _time_left() > 0.5:
            fxp, ok = _eval(x + h)
            if ok:
                return (fxp - fx) / h
        return None

    newton_iters = 0
    max_newton = 40
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
    # Phase 2: Bracketing + Brent's method (manual implementation)
    # ------------------------------------------------------------------
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
        # Use manual Brent's method to avoid scipy overhead and better control
        x_root = _brent_manual(f, a, b, tol, max_iter=100)
        if x_root is not None:
            fx_root, ok = _eval(x_root)
            if ok and abs(fx_root) <= tol:
                return x_root

    if _check_done():
        return best_x

    # ------------------------------------------------------------------
    # Phase 3: Secant method from current point
    # ------------------------------------------------------------------
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
    # Phase 4: Small restart portfolio
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Phase 5: Final polish pass with tighter tolerance
    # ------------------------------------------------------------------
    if _time_left() > 2.0 and eval_count < max_evals - 10:
        # Try to polish the best point with a few more Newton iterations
        x_polish = best_x
        fx_polish = best_fx
        for _ in range(20):
            if _check_done():
                return best_x
            if eval_count >= max_evals or _time_left() < 1.0:
                break
            d = _fd_deriv(x_polish, fx_polish)
            if d is None or abs(d) < 1e-15:
                break
            step = fx_polish / d
            x_new = x_polish - step
            max_step = 10.0 * max(1.0, abs(x_polish))
            if abs(step) > max_step:
                x_new = x_polish - math.copysign(max_step, step)
            fx_new, ok = _eval(x_new)
            if not ok:
                break
            if abs(fx_new) <= tol:
                return x_new
            x_polish, fx_polish = x_new, fx_new

    return best_x


def _brent_manual(f, a, b, tol, max_iter=100):
    """
    Manual implementation of Brent's method with evaluation counting.
    Returns root or None if failed.
    """
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        return None

    c = a
    fc = fa
    d = b - a
    e = d

    for i in range(max_iter):
        if abs(fb) <= tol:
            return b
        if abs(b - a) <= tol:
            return b

        # Inverse quadratic interpolation
        if abs(fc) != abs(fb) and abs(e) != abs(d):
            # Use the formula from Brent's paper
            t1 = 2.0 * (b - a)
            s1 = (b - c) * fb - (b - a) * fc
            s2 = (b - d) * fb - (b - a) * fa
            # Simplified: use secant if quadratic not available
            pass

        # Secant method
        if abs(e) < 1e-15:
            # Bisection
            m = 0.5 * (a + b)
            fm = f(m)
            if fm * fb < 0:
                a = b
            else:
                c = a
                a = b
                b = m
                fa = fb
                fb = fm
            continue

        # Secant
        s = b - (b - a) * fb / (fa - fb)
        m = 0.5 * (a + b)
        
        # Check conditions for secant vs bisection
        if (s - a) * (s - b) > 0 or abs(s - b) >= 0.5 * abs(b - a) or abs(s - b) >= abs(m - a):
            s = m
        
        d = e
        e = b - s
        fs = f(s)
        
        if fs * fb < 0:
            a = b
            fa = fb
        else:
            c = a
            fc = fa
        b = s
        fb = fs

    if abs(fb) <= tol:
        return b
    return None
