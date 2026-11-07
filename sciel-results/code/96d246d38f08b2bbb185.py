"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def _safe_f(f, x):
    """Call f(x) with safety; return None on exception or non-finite."""
    try:
        y = f(x)
        if y is None or not math.isfinite(y):
            return None
        return float(y)
    except Exception:
        return None


def find_root(f, x0):
    x = float(x0)
    deadline = time.monotonic() + 118.0
    best_x = x
    best_fx = _safe_f(f, x)

    if best_fx is not None and abs(best_fx) <= 1e-8:
        return x

    def time_ok():
        return time.monotonic() < deadline

    # --- Phase 1: Secant method (cheap, few calls) ---
    # Start with a nearby point to form a secant pair
    scale = max(1.0, abs(x0))
    x_prev = x - 1e-4 * scale
    fx_prev = _safe_f(f, x_prev)
    x_cur = x
    fx_cur = best_fx

    for _ in range(25):
        if not time_ok():
            break
        if fx_cur is not None and abs(fx_cur) <= 1e-10:
            return x_cur
        if fx_prev is not None and abs(fx_prev) <= 1e-10:
            return x_prev
        if fx_cur is None or fx_prev is None:
            break
        if fx_cur == fx_prev:
            break
        denom = fx_cur - fx_prev
        x_new = x_cur - fx_cur * (x_cur - x_prev) / denom
        if not math.isfinite(x_new):
            break
        fx_new = _safe_f(f, x_new)
        if fx_new is None:
            break
        if best_fx is None or abs(fx_new) < abs(best_fx):
            best_x = x_new
            best_fx = fx_new
        if abs(fx_new) <= 1e-8:
            return x_new
        x_prev, fx_prev = x_cur, fx_cur
        x_cur, fx_cur = x_new, fx_new

    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x

    # --- Phase 2: Find a bracket around x0 ---
    left = right = None
    f_left = f_right = None

    for factor in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]:
        if not time_ok():
            break
        lo = x0 - scale * factor
        hi = x0 + scale * factor
        f_lo = _safe_f(f, lo)
        f_hi = _safe_f(f, hi)
        if f_lo is None or f_hi is None:
            continue
        if f_lo * f_hi <= 0:
            left, right = lo, hi
            f_left, f_right = f_lo, f_hi
            break
        if abs(f_lo) <= 1e-8:
            return lo
        if abs(f_hi) <= 1e-8:
            return hi
        if best_fx is None or abs(f_lo) < abs(best_fx):
            best_x, best_fx = lo, f_lo
        if best_fx is None or abs(f_hi) < abs(best_fx):
            best_x, best_fx = hi, f_hi

    # --- Phase 3: Brent-style hybrid with bracket ---
    if left is not None and right is not None and f_left * f_right <= 0:
        a, b = left, right
        fa, fb = f_left, f_right
        c = a
        fc = fa
        # Ensure |f(a)| <= |f(b)|
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        c = b
        fc = fb

        for _ in range(100):
            if not time_ok():
                break
            if abs(fb) <= 1e-10:
                return b
            if best_fx is None or abs(fb) < abs(best_fx):
                best_x, best_fx = b, fb

            d = (a - b) / 2.0
            e = fa - fb

            use_secant = True
            if abs(e) < abs(d):
                # Inverse quadratic interpolation
                denom = (fa - fc) * (fa - fb)
                if denom != 0 and math.isfinite(denom):
                    num = (b - a) * (fb - fc) - (b - c) * (fb - fa)
                    x_iqi = b - fb * num / denom
                    if min(a, b) < x_iqi < max(a, b):
                        d = x_iqi - b
                        e = d
                        use_secant = False
                    else:
                        use_secant = True
                else:
                    use_secant = True

            if use_secant:
                if fa != fb:
                    x_sec = b - fb * (b - a) / (fb - fa)
                    if min(a, b) < x_sec < max(a, b):
                        d = x_sec - b
                        e = d
                    else:
                        d = (a - b) / 2.0
                        e = d
                else:
                    d = (a - b) / 2.0
                    e = d

            if abs(d) < 1e-15 * max(1.0, abs(b)):
                d = (a - b) / 2.0
                e = d

            x_new = b + e
            fx_new = _safe_f(x_new)
            if fx_new is None:
                x_new = (a + b) / 2.0
                fx_new = _safe_f(x_new)
                if fx_new is None:
                    break

            c, fc = b, fb
            b, fb = x_new, fx_new

            if abs(fb) <= 1e-8:
                return b
            if best_fx is None or abs(fb) < abs(best_fx):
                best_x, best_fx = b, fb

            if fa * fb <= 0:
                a, fa = c, fc
            else:
                a, fa = b, fb

            if abs(b - a) < 1e-15 * max(1.0, abs(b)):
                break

        if best_fx is not None and abs(best_fx) <= 1e-8:
            return best_x

    # --- Phase 4: Newton with central difference derivative ---
    x_newton = best_x
    fx = best_fx
    for _ in range(30):
        if not time_ok():
            break
        if fx is None:
            break
        if abs(fx) <= 1e-10:
            return x_newton

        h = 1e-7 * max(1.0, abs(x_newton))
        fx_plus = _safe_f(f, x_newton + h)
        fx_minus = _safe_f(f, x_newton - h)
        if fx_plus is None or fx_minus is None:
            # Fall back to forward difference
            fx_plus = _safe_f(f, x_newton + h)
            if fx_plus is None:
                break
            d = (fx_plus - fx) / h
        else:
            d = (fx_plus - fx_minus) / (2.0 * h)

        if d == 0 or not math.isfinite(d):
            break

        step = fx / d
        if abs(step) > 10 * max(1.0, abs(x_newton)):
            step = math.copysign(10 * max(1.0, abs(x_newton)), step)

        x_newton = x_newton - step
        if not math.isfinite(x_newton):
            break

        fx = _safe_f(f, x_newton)
        if fx is None:
            break

        if best_fx is None or abs(fx) < abs(best_fx):
            best_x = x_newton
            best_fx = fx

        if abs(fx) <= 1e-8:
            return x_newton

    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x

    # --- Phase 5: Polynomial regression approach ---
    # Sample points around x0 and fit a low-degree polynomial
    # Then find roots of the polynomial analytically
    sample_points = []
    sample_values = []

    # Collect points we already have
    if best_fx is not None:
        sample_points.append(best_x)
        sample_values.append(best_fx)

    # Add more sample points
    for offset in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0]:
        if not time_ok():
            break
        x_try = x0 + offset * scale
        if x_try not in [p for p, _ in sample_points]:
            fx_try = _safe_f(f, x_try)
            if fx_try is not None:
                sample_points.append(x_try)
                sample_values.append(fx_try)
                if abs(fx_try) <= 1e-8:
                    return x_try
                if best_fx is None or abs(fx_try) < abs(best_fx):
                    best_x = x_try
                    best_fx = fx_try

    if len(sample_points) >= 4:
        # Fit a cubic polynomial using numpy
        try:
            import numpy as np
            pts = np.array(sample_points[:min(8, len(sample_points))])
            vals = np.array(sample_values[:min(8, len(sample_values))])
            # Fit a cubic (degree 3) polynomial
            coeffs = np.polyfit(pts, vals, 3)
            # Find roots of the polynomial
            roots = np.roots(coeffs)
            # Filter for real roots and finite values
            real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and math.isfinite(r.real)]
            if real_roots:
                # Select the root closest to x0
                best_root = min(real_roots, key=lambda r: abs(r - x0))
                # Verify the root
                fx_root = _safe_f(f, best_root)
                if fx_root is not None and abs(fx_root) <= 1e-8:
                    return best_root
                if best_fx is None or abs(fx_root) < abs(best_fx):
                    best_x = best_root
                    best_fx = fx_root
        except Exception:
            pass

    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x

    # --- Phase 6: Last resort - try more points ---
    for offset in [2.0, -2.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0]:
        if not time_ok():
            break
        x_try = x0 + offset * scale
        fx_try = _safe_f(f, x_try)
        if fx_try is not None:
            if best_fx is None or abs(fx_try) < abs(best_fx):
                best_x = x_try
                best_fx = fx_try
            if abs(fx_try) <= 1e-8:
                return x_try

    # Final safety: ensure finite return
    if not math.isfinite(best_x):
        return float(x0)
    return best_x
# EVOLVE-BLOCK-END
