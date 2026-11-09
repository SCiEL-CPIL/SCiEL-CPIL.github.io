"""Root finder for 3 fixed oscillatory equations with optimized call budget."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x with |f(x)| <= 1e-8.

    Strategy:
    1. Local bracketing search with geometric expansion (10x per level).
    2. Hybrid Secant/Bisection convergence.
    3. Final high-precision polish pass if close.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    tol = 1e-8

    x0 = float(x0)

    # 1. Initial Evaluation
    try:
        f0 = f(x0)
    except Exception:
        x_try = x0 + 1e-6
        try:
            f0 = f(x_try)
            x0 = x_try
        except Exception:
            return x0  # Last resort

    if not math.isfinite(f0):
        return x0

    if abs(f0) <= tol:
        return x0

    # 2. Bracketing Search via geometric expansion
    # Start small and expand by factor 10 each level.
    # This covers [1e-4, 1e3] in 8 levels with 16 evaluations max,
    # vs 6 fixed deltas with 12 evaluations but less adaptive coverage.
    a = None
    b = None
    fa = None
    fb = None

    best_x = x0
    best_f = f0
    best_abs = abs(f0)

    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs
        if math.isfinite(x) and math.isfinite(fx):
            af = abs(fx)
            if af < best_abs:
                best_x = x
                best_f = fx
                best_abs = af

    # Geometric expansion: d starts at 1e-4, multiplies by 10 each iteration
    d = 1e-4
    max_d = 1e6  # Cap to avoid infinite loop
    while d <= max_d:
        if time.time() > deadline:
            break

        x_left = x0 - d
        x_right = x0 + d

        # Evaluate left
        try:
            fl = f(x_left)
        except Exception:
            fl = float('nan')

        if math.isfinite(fl):
            update_best(x_left, fl)
            if abs(fl) <= tol:
                return x_left
            # Check sign change with x0
            if f0 * fl < 0:
                a, fa = x_left, fl
                b, fb = x0, f0
                break

        # Evaluate right
        try:
            fr = f(x_right)
        except Exception:
            fr = float('nan')

        if math.isfinite(fr):
            update_best(x_right, fr)
            if abs(fr) <= tol:
                return x_right
            # Check sign change with x0
            if f0 * fr < 0:
                a, fa = x0, f0
                b, fb = x_right, fr
                break

        # Also check sign change between left and right (both sides of x0)
        if math.isfinite(fl) and math.isfinite(fr) and fl * fr < 0:
            a, fa = x_left, fl
            b, fb = x_right, fr
            break

        d *= 10.0

    # If we found a bracket, use Hybrid Secant/Bisection
    if a is not None and b is not None:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Hybrid Iteration
        for _ in range(50):
            if time.time() > deadline:
                break

            if abs(fa) <= tol:
                return a
            if abs(fb) <= tol:
                return b

            # Calculate Secant Point
            denom = fb - fa
            if denom == 0 or not math.isfinite(denom):
                x_new = (a + b) / 2
            else:
                x_secant = b - fb * (b - a) / denom
                if a < x_secant < b:
                    x_new = x_secant
                else:
                    x_new = (a + b) / 2

            if not math.isfinite(x_new):
                break

            try:
                f_new = f(x_new)
            except Exception:
                break

            if not math.isfinite(f_new):
                break

            update_best(x_new, f_new)

            if abs(f_new) <= tol:
                return x_new

            # Update Bracket
            if fa * f_new < 0:
                b, fb = x_new, f_new
            else:
                a, fa = x_new, f_new

            if abs(b - a) < 1e-15 * max(1.0, abs(b)):
                break

        # Polish pass
        if best_abs <= 1e-6:
            if a is not None and b is not None and min(a, b) <= best_x <= max(a, b):
                pa, pb = a, b
                pfa, pfb = fa, fb
            else:
                eps = 1e-10
                pa, pb = best_x - eps, best_x + eps
                try:
                    pfa = f(pa)
                    pfb = f(pb)
                except Exception:
                    pfa, pfb = float('nan'), float('nan')

                if not (math.isfinite(pfa) and math.isfinite(pfb)):
                    return best_x

            if math.isfinite(pfa) and math.isfinite(pfb) and pfa * pfb < 0:
                for _ in range(25):
                    if time.time() > deadline:
                        break
                    if abs(pfa) <= tol:
                        return pa
                    if abs(pfb) <= tol:
                        return pb

                    x_mid = (pa + pb) / 2
                    try:
                        f_mid = f(x_mid)
                    except Exception:
                        break
                    if not math.isfinite(f_mid):
                        break
                    update_best(x_mid, f_mid)
                    if abs(f_mid) <= tol:
                        return x_mid

                    if pfa * f_mid < 0:
                        pb, pfb = x_mid, f_mid
                    else:
                        pa, pfa = x_mid, f_mid

        return best_x if best_abs <= tol else (a + b) / 2

    else:
        # No bracket found. Try Secant with available points.
        x1 = x0
        f1 = f0

        # Find a second point
        candidates = [x0 + 1e-2, x0 - 1e-2, x0 + 1.0, x0 - 1.0]

        x2 = None
        f2 = None
        min_abs = float('inf')

        for cx in candidates:
            try:
                cf = f(cx)
            except Exception:
                continue
            if math.isfinite(cf):
                update_best(cx, cf)
                if abs(cf) < min_abs:
                    min_abs = abs(cf)
                    x2 = cx
                    f2 = cf

        if x2 is None:
            x2 = x0 + 10.0
            try:
                f2 = f(x2)
            except Exception:
                return best_x
            if not math.isfinite(f2):
                return best_x
            update_best(x2, f2)

        # Run Secant
        for _ in range(30):
            if time.time() > deadline:
                break

            if abs(f1) <= tol:
                return x1
            if abs(f2) <= tol:
                return x2

            denom = f2 - f1
            if denom == 0 or not math.isfinite(denom):
                break

            x_new = x2 - f2 * (x2 - x1) / denom

            if not math.isfinite(x_new):
                break

            try:
                f_new = f(x_new)
            except Exception:
                break

            if not math.isfinite(f_new):
                break

            update_best(x_new, f_new)

            if abs(f_new) <= tol:
                return x_new

            x1, f1 = x2, f2
            x2, f2 = x_new, f_new

        # Polish pass for Secant case
        if best_abs <= 1e-6:
            eps = 1e-10
            pa, pb = best_x - eps, best_x + eps
            try:
                pfa = f(pa)
                pfb = f(pb)
            except Exception:
                return best_x

            if math.isfinite(pfa) and math.isfinite(pfb) and pfa * pfb < 0:
                for _ in range(25):
                    if time.time() > deadline:
                        break
                    if abs(pfa) <= tol:
                        return pa
                    if abs(pfb) <= tol:
                        return pb

                    x_mid = (pa + pb) / 2
                    try:
                        f_mid = f(x_mid)
                    except Exception:
                        break
                    if not math.isfinite(f_mid):
                        break
                    update_best(x_mid, f_mid)
                    if abs(f_mid) <= tol:
                        return x_mid

                    if pfa * f_mid < 0:
                        pb, pfb = x_mid, f_mid
                    else:
                        pa, pfa = x_mid, f_mid

        return best_x

# EVOLVE-BLOCK-END
