"""Root finder for smooth polynomial equations - safeguarded Newton-Raphson."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit

    x = float(x0)

    # Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # Try nearby points to find a finite value
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                xt = x + sign * dx
                try:
                    fxt = f(xt)
                except Exception:
                    continue
                if math.isfinite(fxt):
                    x = xt
                    fx = fxt
                    break
            if math.isfinite(fx):
                break
        if not math.isfinite(fx):
            return x0

    if abs(fx) <= 1e-8:
        return x

    best_x = x
    best_fx = fx
    best_abs = abs(fx)

    # Safeguarded Newton-Raphson
    # We need a derivative estimate. Use finite differences.
    # Strategy: maintain x, fx, and a derivative estimate dfx.
    # At each step:
    #   1. Compute Newton step: x_new = x - fx/dfx
    #   2. If step is too large, shrink it (damping)
    #   3. Evaluate f(x_new)
    #   4. Update derivative estimate using the new point
    #   5. If |f(x_new)| < best, update best
    #   6. If |f(x_new)| <= 1e-8, return
    #   7. If we detect a sign change, we can use bisection as fallback

    # Initial derivative estimate via central difference
    h0 = max(1e-6, abs(x) * 1e-6)
    dfx = None
    try:
        fx_plus = f(x + h0)
        fx_minus = f(x - h0)
        if math.isfinite(fx_plus) and math.isfinite(fx_minus):
            dfx = (fx_plus - fx_minus) / (2 * h0)
        elif math.isfinite(fx_plus):
            dfx = (fx_plus - fx) / h0
        elif math.isfinite(fx_minus):
            dfx = (fx - fx_minus) / h0
    except Exception:
        pass

    if dfx is None or not math.isfinite(dfx) or abs(dfx) < 1e-14:
        # Fallback: try forward difference with different h
        for h in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]:
            try:
                fx_h = f(x + h)
                if math.isfinite(fx_h):
                    d = (fx_h - fx) / h
                    if math.isfinite(d) and abs(d) >= 1e-14:
                        dfx = d
                        break
            except Exception:
                continue

    if dfx is None or not math.isfinite(dfx) or abs(dfx) < 1e-14:
        # Can't get a derivative; fall back to secant-like exploration
        # Try a few points to find one with smaller |f|
        for h in [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                xt = x + sign * h
                try:
                    fxt = f(xt)
                except Exception:
                    continue
                if not math.isfinite(fxt):
                    continue
                if abs(fxt) < best_abs:
                    best_x = xt
                    best_fx = fxt
                    best_abs = abs(fxt)
                if abs(fxt) <= 1e-8:
                    return xt
        return best_x

    # Newton-Raphson iterations with safeguards
    max_iter = 50
    damping = 1.0
    x_prev = x
    fx_prev = fx

    for i in range(max_iter):
        if time.time() > deadline:
            break

        if abs(fx) <= 1e-8:
            return x

        # Newton step
        step = fx / dfx
        x_new = x - step

        # Safeguard: limit step size
        max_step = max(1.0, abs(x))
        if abs(step) > max_step * damping:
            step = math.copysign(max_step * damping, step)
            x_new = x - step

        # Try the candidate
        try:
            f_new = f(x_new)
        except Exception:
            # Shrink and retry
            damping *= 0.5
            if damping < 1e-6:
                break
            continue

        if not math.isfinite(f_new):
            damping *= 0.5
            if damping < 1e-6:
                break
            continue

        if abs(f_new) < best_abs:
            best_x = x_new
            best_fx = f_new
            best_abs = abs(f_new)

        if abs(f_new) <= 1e-8:
            return x_new

        # Check for sign change between x and x_new
        sign_change = (fx > 0 and f_new < 0) or (fx < 0 and f_new > 0)

        # Update derivative estimate
        # Use the two most recent points for a secant-based derivative
        if abs(x_new - x_prev) > 1e-14:
            dfx_new = (f_new - fx_prev) / (x_new - x_prev)
            if math.isfinite(dfx_new) and abs(dfx_new) >= 1e-14:
                # Blend with previous derivative for stability
                dfx = 0.5 * dfx + 0.5 * dfx_new
            else:
                # Keep old derivative
                pass
        else:
            # Points too close; use finite difference at new point
            h = max(1e-6, abs(x_new) * 1e-6)
            try:
                fp = f(x_new + h)
                if math.isfinite(fp):
                    d = (fp - f_new) / h
                    if math.isfinite(d) and abs(d) >= 1e-14:
                        dfx = d
            except Exception:
                pass

        # If sign change, we have a bracket [x, x_new]
        # Use bisection to be safe
        if sign_change:
            a = min(x, x_new)
            b = max(x, x_new)
            # Do a few bisection steps to narrow the bracket
            for _ in range(20):
                if time.time() > deadline:
                    break
                mid = 0.5 * (a + b)
                try:
                    f_mid = f(mid)
                except Exception:
                    break
                if not math.isfinite(f_mid):
                    break
                if abs(f_mid) < best_abs:
                    best_x = mid
                    best_fx = f_mid
                    best_abs = abs(f_mid)
                if abs(f_mid) <= 1e-8:
                    return mid
                if (f_mid > 0 and fx > 0) or (f_mid < 0 and fx < 0):
                    a = mid
                else:
                    b = mid
                if abs(b - a) < 1e-12:
                    break
            # After bisection, return best
            return best_x

        # Update state
        x_prev = x
        fx_prev = fx
        x = x_new
        fx = f_new

        # Reset damping if we made progress
        if abs(fx) < abs(fx_prev):
            damping = min(1.0, damping * 2.0)
        else:
            damping *= 0.7

    return best_x
# EVOLVE-BLOCK-END
