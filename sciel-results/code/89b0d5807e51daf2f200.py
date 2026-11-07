"""Root finder for smooth polynomial equations - safeguarded Newton-Raphson hybrid."""
import math
import time


def find_root(f, x0):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    def elapsed_ok():
        return time.monotonic() < deadline

    x = float(x0)

    # Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                xt = x + sign * dx
                try:
                    fxt = f(xt)
                except Exception:
                    continue
                if math.isfinite(fxt):
                    x, fx = xt, fxt
                    break
            if math.isfinite(fx):
                break
        if not math.isfinite(fx):
            return x0

    if abs(fx) <= 1e-8:
        return x

    best_x, best_f = x, fx

    # Phase 1: Quick bracket search (limited to save calls)
    a, b = x, x
    fa, fb = fx, fx
    step = 1.0
    bracketed = False

    for i in range(10):
        if not elapsed_ok():
            break
        a = x - step
        b = x + step
        try:
            fa = f(a)
            fb = f(b)
        except Exception:
            step *= 2.0
            continue

        if not (math.isfinite(fa) and math.isfinite(fb)):
            step *= 2.0
            continue

        if abs(fa) < abs(best_f):
            best_x, best_f = a, fa
        if abs(fb) < abs(best_f):
            best_x, best_f = b, fb

        if fa == 0.0:
            return a
        if fb == 0.0:
            return b
        if fa * fb < 0.0:
            bracketed = True
            break
        step *= 2.0

    if bracketed:
        # Use safeguarded Newton with bisection fallback
        lo, hi = a, b
        flo, fhi = fa, fb
        if abs(flo) > abs(fhi):
            lo, hi = hi, lo
            flo, fhi = fhi, flo

        x_curr = lo
        f_curr = flo

        for it in range(60):
            if not elapsed_ok():
                break

            if abs(f_curr) <= 1e-8:
                return x_curr

            # Finite difference derivative
            h = 1e-8 * max(1.0, abs(x_curr))
            try:
                f_xh = f(x_curr + h)
            except Exception:
                break
            if not math.isfinite(f_xh):
                break

            d = (f_xh - f_curr) / h
            if d == 0 or not math.isfinite(d):
                break

            # Newton step
            x_new = x_curr - f_curr / d

            # Safeguard: stay within bracket
            if x_new <= lo or x_new >= hi:
                x_new = 0.5 * (lo + hi)

            try:
                f_new = f(x_new)
            except Exception:
                x_new = 0.5 * (lo + hi)
                try:
                    f_new = f(x_new)
                except Exception:
                    break
            if not math.isfinite(f_new):
                x_new = 0.5 * (lo + hi)
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(f_new):
                    break

            if abs(f_new) <= 1e-8:
                return x_new

            if abs(f_new) < abs(best_f):
                best_x, best_f = x_new, f_new

            # Update bracket
            if f_new * flo < 0:
                hi, fhi = x_new, f_new
            else:
                lo, flo = x_new, f_new

            x_curr = x_new
            f_curr = f_new

        return best_x
    else:
        # No bracket: safeguarded Newton-Raphson minimizing |f(x)|
        x_curr = x
        f_curr = fx

        # Get a second point for initial derivative estimate
        h = 1e-7 * max(1.0, abs(x_curr))
        try:
            f_xh = f(x_curr + h)
        except Exception:
            f_xh = None

        if f_xh is not None and math.isfinite(f_xh):
            if abs(f_xh) < abs(best_f):
                best_x, best_f = x_curr + h, f_xh
            if abs(f_xh) <= 1e-8:
                return x_curr + h

        for it in range(80):
            if not elapsed_ok():
                break

            if abs(f_curr) <= 1e-8:
                return x_curr

            # Finite difference derivative
            h = 1e-7 * max(1.0, abs(x_curr))
            try:
                f_xh = f(x_curr + h)
            except Exception:
                break
            if not math.isfinite(f_xh):
                break

            d = (f_xh - f_curr) / h
            if d == 0 or not math.isfinite(d):
                break

            # Newton step
            x_new = x_curr - f_curr / d

            # Safeguard: limit step size
            max_step = max(1.0, abs(x_curr)) * 5.0
            if abs(x_new - x_curr) > max_step:
                x_new = x_curr + max_step * (1.0 if x_new > x_curr else -1.0)

            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break

            if abs(f_new) < abs(f_curr):
                x_curr, f_curr = x_new, f_new
                if abs(f_curr) < abs(best_f):
                    best_x, best_f = x_curr, f_curr
            else:
                # Damped step: try halfway
                x_half = 0.5 * (x_curr + x_new)
                try:
                    f_half = f(x_half)
                except Exception:
                    break
                if not math.isfinite(f_half):
                    break

                if abs(f_half) < abs(f_curr):
                    x_curr, f_curr = x_half, f_half
                    if abs(f_curr) < abs(best_f):
                        best_x, best_f = x_curr, f_curr
                else:
                    # Try smaller step
                    x_quarter = 0.75 * x_curr + 0.25 * x_new
                    try:
                        f_quarter = f(x_quarter)
                    except Exception:
                        break
                    if not math.isfinite(f_quarter):
                        break

                    if abs(f_quarter) < abs(f_curr):
                        x_curr, f_curr = x_quarter, f_quarter
                        if abs(f_curr) < abs(best_f):
                            best_x, best_f = x_curr, f_curr
                    else:
                        break

        # Final polish with tighter tolerance
        if abs(best_f) > 1e-8 and abs(best_f) < 1.0:
            x_curr = best_x
            f_curr = best_f
            for it in range(30):
                if not elapsed_ok():
                    break
                if abs(f_curr) <= 1e-8:
                    return x_curr

                h = 1e-9 * max(1.0, abs(x_curr))
                try:
                    f_xh = f(x_curr + h)
                except Exception:
                    break
                if not math.isfinite(f_xh):
                    break

                d = (f_xh - f_curr) / h
                if d == 0 or not math.isfinite(d):
                    break

                x_new = x_curr - f_curr / d

                max_step = max(1.0, abs(x_curr)) * 0.1
                if abs(x_new - x_curr) > max_step:
                    x_new = x_curr + max_step * (1.0 if x_new > x_curr else -1.0)

                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(f_new):
                    break

                if abs(f_new) < abs(f_curr):
                    x_curr, f_curr = x_new, f_new
                    if abs(f_curr) < abs(best_f):
                        best_x, best_f = x_curr, f_curr
                else:
                    break

    return best_x
