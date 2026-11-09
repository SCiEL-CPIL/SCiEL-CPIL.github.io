"""Root finder for smooth polynomial equations with call-budget awareness."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s kill
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using a combination of:
    1. Secant method (1 call per iteration after the first two)
    2. Bisection fallback for robustness
    3. Newton polish at the end for high precision
    4. Deadline guard to stay within wall-clock limit
    """
    x = float(x0)
    if not _is_finite(x):
        x = 0.0

    # Track best result seen so far
    best_x = x
    best_fx = None

    def safe_eval(val):
        """Evaluate f at val, return (fx, is_finite)."""
        try:
            fx = f(val)
            if fx is None:
                return None, False
            fx = float(fx)
            if not math.isfinite(fx):
                return None, False
            return fx, True
        except Exception:
            return None, False

    # --- Phase 1: Secant method with bisection safeguard ---
    # We maintain two points (x_a, f_a) and (x_b, f_b)
    # If f_a and f_b have opposite signs, we can bisect
    # Otherwise, we use secant step

    x_a = x
    fx_a, ok = safe_eval(x_a)
    if not ok:
        # Try a small perturbation
        x_a = x + 1e-4
        fx_a, ok = safe_eval(x_a)
        if not ok:
            x_a = x - 1e-4
            fx_a, ok = safe_eval(x_a)
            if not ok:
                return x_a  # give up, return best we have

    if abs(fx_a) <= 1e-8:
        return x_a

    best_x = x_a
    best_fx = fx_a

    # Second point for secant
    h = 1e-4 * max(1.0, abs(x_a))
    x_b = x_a + h
    fx_b, ok = safe_eval(x_b)
    if not ok:
        x_b = x_a - h
        fx_b, ok = safe_eval(x_b)
        if not ok:
            return best_x

    if abs(fx_b) <= 1e-8:
        return x_b

    if abs(fx_b) < abs(best_fx):
        best_x = x_b
        best_fx = fx_b

    # Main loop: secant with bisection safeguard
    max_iter = 50
    for i in range(max_iter):
        if _time_left() < 2.0:
            break

        # Check if we already have a good enough answer
        if abs(best_fx) <= 1e-8:
            break

        # Determine step
        if fx_a * fx_b < 0:
            # Opposite signs: use bisection for robustness
            x_mid = (x_a + x_b) / 2.0
            fx_mid, ok = safe_eval(x_mid)
            if not ok:
                # Fallback to secant
                if fx_b != fx_a:
                    x_new = x_a - fx_a * (x_b - x_a) / (fx_b - fx_a)
                else:
                    x_new = (x_a + x_b) / 2.0
            else:
                x_new = x_mid
                fx_new = fx_mid
        else:
            # Same sign: use secant
            if fx_b != fx_a:
                x_new = x_a - fx_a * (x_b - x_a) / (fx_b - fx_a)
            else:
                # Degenerate: perturb
                x_new = x_a + 1e-6 * max(1.0, abs(x_a))
                fx_new, ok = safe_eval(x_new)
                if not ok:
                    break
                # Update and continue
                x_a, fx_a = x_new, fx_new
                continue

        if not _is_finite(x_new):
            break

        # Clamp x_new to reasonable range
        if abs(x_new) > 1e10:
            break

        fx_new, ok = safe_eval(x_new)
        if not ok:
            break

        if abs(fx_new) < abs(best_fx):
            best_x = x_new
            best_fx = fx_new

        if abs(fx_new) <= 1e-8:
            return x_new

        # Update points: keep the point with smaller |f| as x_a,
        # and the new point as x_b, but ensure we maintain bracket if possible
        if fx_a * fx_new < 0:
            # New point brackets with x_a
            x_b = x_new
            fx_b = fx_new
        elif fx_b * fx_new < 0:
            # New point brackets with x_b
            x_a = x_new
            fx_a = fx_new
        else:
            # No bracket: keep the two points with smallest |f|
            if abs(fx_new) < abs(fx_a):
                x_a, fx_a = x_new, fx_new
            else:
                x_b, fx_b = x_new, fx_new

    # --- Phase 2: Newton polish ---
    # Use finite-difference Newton for a few iterations to tighten
    x = best_x
    for i in range(20):
        if _time_left() < 1.0:
            break

        fx, ok = safe_eval(x)
        if not ok:
            break

        if abs(fx) <= 1e-8:
            return x

        h = 1e-7 * max(1.0, abs(x))
        fx_plus, ok = safe_eval(x + h)
        if not ok:
            break

        d = (fx_plus - fx) / h
        if d == 0 or not math.isfinite(d):
            break

        x_new = x - fx / d
        if not _is_finite(x_new):
            break

        # Damping: if step is too large, halve it
        step = abs(x_new - x)
        if step > 10 * max(1.0, abs(x)):
            x_new = x + 0.5 * (x_new - x) * math.copysign(1.0, x_new - x)

        x = x_new

        fx_check, ok = safe_eval(x)
        if ok and abs(fx_check) < abs(best_fx):
            best_x = x
            best_fx = fx_check
        if ok and abs(fx_check) <= 1e-8:
            return x

    return best_x
