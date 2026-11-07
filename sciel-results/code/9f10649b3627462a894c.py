"""Combined root finder: Newton with forward-diff (call-efficient) + bracketing fallback + polish."""
import math
import time

_DEADLINE = time.monotonic() + 140.0  # safety margin before 150s kill


def _check_deadline():
    return time.monotonic() < _DEADLINE


def find_root(f, x0):
    x = float(x0)
    best_x = x
    best_fx = None

    # Evaluate initial
    try:
        fx = f(x)
    except Exception:
        return x
    if not math.isfinite(fx):
        return x

    if abs(fx) <= 1e-8:
        return x

    best_fx = fx
    best_x = x

    prev_x = x
    prev_fx = fx

    # Phase 1: Newton's method with forward-difference derivative (1 extra call per iter)
    converged = False
    for i in range(50):
        if not _check_deadline():
            break
        if abs(fx) <= 1e-8:
            return x

        h = 1e-7 * max(1.0, abs(x))
        try:
            fxh = f(x + h)
        except Exception:
            break
        if not math.isfinite(fxh):
            break

        d = (fxh - fx) / h

        if abs(d) < 1e-30:
            # Derivative near zero, try secant step
            if abs(prev_x - x) > 1e-15:
                d_sec = (fx - prev_fx) / (x - prev_x)
                if abs(d_sec) > 1e-30:
                    x_new = x - fx / d_sec
                else:
                    x_new = x + 1e-6
            else:
                x_new = x + 1e-6
        else:
            x_new = x - fx / d
            # Damping: if step is too large, reduce it
            if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
                x_new = x + 10.0 * max(1.0, abs(x)) * (1 if x_new > x else -1)

        if not math.isfinite(x_new):
            break

        prev_x = x
        prev_fx = fx
        x = x_new

        try:
            fx = f(x)
        except Exception:
            break
        if not math.isfinite(fx):
            break

        if abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx

        if abs(fx) <= 1e-8:
            converged = True
            break

    if converged:
        return x

    # Phase 2: If Newton didn't converge, try bracketing + bisection/interpolation
    if abs(best_fx) > 1e-8 and _check_deadline():
        x_bracketed = _bracket_and_solve(f, best_x, best_fx)
        if x_bracketed is not None:
            try:
                fx_br = f(x_bracketed)
                if math.isfinite(fx_br) and abs(fx_br) < abs(best_fx):
                    best_x = x_bracketed
                    best_fx = fx_br
            except Exception:
                pass

    # Phase 3: High-precision polish if close
    if best_fx is not None and abs(best_fx) <= 1e-6 and _check_deadline():
        x = _polish(f, best_x, best_fx)
        try:
            fx_check = f(x)
            if math.isfinite(fx_check) and abs(fx_check) < abs(best_fx):
                best_x = x
                best_fx = fx_check
        except Exception:
            pass

    return best_x


def _polish(f, x, fx):
    """High-precision Newton polish with central difference and tighter tolerance."""
    for _ in range(20):
        if not _check_deadline():
            break
        if not math.isfinite(x) or fx is None or not math.isfinite(fx):
            break
        if abs(fx) <= 1e-12:
            break

        h = 1e-8 * max(1.0, abs(x))
        try:
            f_xp = f(x + h)
            f_xm = f(x - h)
        except Exception:
            break
        if not math.isfinite(f_xp) or not math.isfinite(f_xm):
            break

        d = (f_xp - f_xm) / (2.0 * h)
        if abs(d) < 1e-16:
            break

        x_new = x - fx / d
        if not math.isfinite(x_new):
            break

        # Small damping for polish
        step = x_new - x
        if abs(step) > 0.1:
            x_new = x + (0.1 if step > 0 else -0.1)

        x = x_new
        try:
            fx = f(x)
        except Exception:
            break
        if not math.isfinite(fx):
            break

    return x


def _bracket_and_solve(f, x_start, fx_start):
    """Try to find a bracket and solve using bisection/interpolation."""
    if not _check_deadline():
        return None
    if fx_start is None or not math.isfinite(fx_start):
        return None

    a = x_start
    fa = fx_start
    step = 1.0
    b = None
    fb = None

    # Expand to find a sign change
    for _ in range(30):
        if not _check_deadline():
            break
        b = a + step
        try:
            fb = f(b)
        except Exception:
            step *= 2.0
            continue
        if not math.isfinite(fb):
            step *= 2.0
            continue
        if fa * fb <= 0:
            break
        # If no sign change, try the other direction
        b = a - step
        try:
            fb = f(b)
        except Exception:
            step *= 2.0
            continue
        if not math.isfinite(fb):
            step *= 2.0
            continue
        if fa * fb <= 0:
            break
        step *= 2.0
    else:
        return None

    # Now we have a bracket [a, b] with fa * fb <= 0
    # Ensure a is the left endpoint
    if a > b:
        a, b = b, a
        fa, fb = fb, fa

    best_x = a if abs(fa) < abs(fb) else b
    best_f = min(abs(fa), abs(fb))

    # Safeguarded inverse interpolation
    for _ in range(50):
        if not _check_deadline():
            break
        if abs(fb) <= 1e-8:
            return b
        if abs(fa) <= 1e-8:
            return a

        denom = fb - fa
        if abs(denom) < 1e-300:
            x_new = (a + b) / 2.0
        else:
            x_new = (a * fb - b * fa) / denom

        # Safeguard: x_new must be in (a, b)
        if not (a < x_new < b):
            x_new = (a + b) / 2.0

        try:
            f_new = f(x_new)
        except Exception:
            x_new = (a + b) / 2.0
            try:
                f_new = f(x_new)
            except Exception:
                break
        if not math.isfinite(f_new):
            x_new = (a + b) / 2.0
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break

        if abs(f_new) < best_f:
            best_x = x_new
            best_f = abs(f_new)

        if abs(f_new) <= 1e-8:
            return x_new

        # Update bracket
        if fa * f_new <= 0:
            b = x_new
            fb = f_new
        else:
            a = x_new
            fa = f_new

        # Check convergence
        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
            break

    if abs(fb) <= abs(fa):
        return b
    else:
        return a
