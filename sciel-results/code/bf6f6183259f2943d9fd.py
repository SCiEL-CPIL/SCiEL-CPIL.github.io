"""Root finder for smooth polynomial equations - IQI + secant hybrid."""
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

    # Phase 1: Quick bracket search
    a, b = x, x
    fa, fb = fx, fx
    step = 1.0
    bracketed = False

    for i in range(12):
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
        # Use IQI with bisection fallback inside the bracket
        lo, hi = a, b
        flo, fhi = fa, fb
        if abs(flo) > abs(fhi):
            lo, hi = hi, lo
            flo, fhi = fhi, flo

        # Initialize three points for IQI: lo, hi, and midpoint
        mid = 0.5 * (lo + hi)
        try:
            fmid = f(mid)
        except Exception:
            fmid = None

        if fmid is not None and math.isfinite(fmid):
            if abs(fmid) < abs(best_f):
                best_x, best_f = mid, fmid
            if abs(fmid) <= 1e-8:
                return mid
            # Three points: p1=lo, p2=mid, p3=hi
            p1, f1 = lo, flo
            p2, f2 = mid, fmid
            p3, f3 = hi, fhi
            have_three = True
        else:
            # Only two points, use secant
            p1, f1 = lo, flo
            p2, f2 = hi, fhi
            have_three = False

        for it in range(60):
            if not elapsed_ok():
                break

            if abs(f2) <= 1e-8:
                return p2

            # Compute next point using IQI if we have 3 points, else secant
            if have_three:
                # Inverse quadratic interpolation
                # Fit quadratic through (f1,p1), (f2,p2), (f3,p3) and find root
                # Using Lagrange interpolation on x as function of f
                d1 = (f2 - f3) / (f1 - f3) if f1 != f3 else 1.0
                d2 = (f1 - f3) / (f1 - f2) if f1 != f2 else 1.0

                x_iqi = f1 * f2 * d1 / (f1 - f2) + f1 * f3 * d2 / (f1 - f3) + f2 * f3 * (1 - d1 - d2) / (f2 - f3) \
                    if (f1 != f2 and f1 != f3 and f2 != f3) else None

                if x_iqi is None or not math.isfinite(x_iqi):
                    # Fallback to secant
                    denom = f3 - f1
                    if abs(denom) > 1e-14:
                        x_new = p3 - f3 * (p3 - p1) / denom
                    else:
                        x_new = 0.5 * (lo + hi)
                else:
                    x_new = x_iqi

                # Check if IQI point is within bracket
                if x_new <= lo or x_new >= hi:
                    # Use secant instead
                    denom = f3 - f1
                    if abs(denom) > 1e-14:
                        x_new = p3 - f3 * (p3 - p1) / denom
                    else:
                        x_new = 0.5 * (lo + hi)

                # Additional safeguard: if still outside, bisection
                if x_new <= lo or x_new >= hi or not math.isfinite(x_new):
                    x_new = 0.5 * (lo + hi)
            else:
                # Secant with two points
                denom = f2 - f1
                if abs(denom) > 1e-14:
                    x_new = p2 - f2 * (p2 - p1) / denom
                else:
                    x_new = 0.5 * (lo + hi)

                if x_new <= lo or x_new >= hi or not math.isfinite(x_new):
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

            # Update three points: shift and add new
            if have_three:
                p1, f1 = p2, f2
                p2, f2 = p3, f3
                p3, f3 = x_new, f_new
            else:
                p1, f1 = p2, f2
                p2, f2 = x_new, f_new
                # Try to get a third point for IQI
                mid_try = 0.5 * (p1 + p2)
                if abs(mid_try - p1) > 1e-12 and abs(mid_try - p2) > 1e-12:
                    try:
                        f_mid_try = f(mid_try)
                    except Exception:
                        f_mid_try = None
                    if f_mid_try is not None and math.isfinite(f_mid_try):
                        if abs(f_mid_try) < abs(best_f):
                            best_x, best_f = mid_try, f_mid_try
                        if abs(f_mid_try) <= 1e-8:
                            return mid_try
                        # Insert into sorted position
                        if mid_try < p1:
                            p3, f3 = p2, f2
                            p2, f2 = p1, f1
                            p1, f1 = mid_try, f_mid_try
                        elif mid_try < p2:
                            p3, f3 = p2, f2
                            p2, f2 = mid_try, f_mid_try
                        else:
                            p3, f3 = mid_try, f_mid_try
                        have_three = True

        return best_x
    else:
        # No bracket: use secant method to minimize |f(x)|
        x_curr = x
        f_curr = fx

        # Get a second point for initial secant
        h = 1e-6 * max(1.0, abs(x_curr))
        try:
            f_xh = f(x_curr + h)
        except Exception:
            f_xh = None

        if f_xh is not None and math.isfinite(f_xh):
            if abs(f_xh) < abs(best_f):
                best_x, best_f = x_curr + h, f_xh
            if abs(f_xh) <= 1e-8:
                return x_curr + h
            x_prev = x_curr + h
            f_prev = f_xh
        else:
            # Try other offsets
            found = False
            for h_try in [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0]:
                for sign in [1, -1]:
                    xt = x_curr + sign * h_try
                    try:
                        fxt = f(xt)
                    except Exception:
                        continue
                    if math.isfinite(fxt):
                        if abs(fxt) < abs(best_f):
                            best_x, best_f = xt, fxt
                        if abs(fxt) <= 1e-8:
                            return xt
                        x_prev = xt
                        f_prev = fxt
                        found = True
                        break
                if found:
                    break
            if not found:
                return best_x

        for it in range(80):
            if not elapsed_ok():
                break

            if abs(f_curr) <= 1e-8:
                return x_curr

            # Secant step
            denom = f_curr - f_prev
            if abs(denom) < 1e-14:
                # Nearly parallel, perturb
                x_new = x_curr + 1e-6
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(f_new):
                    break
                if abs(f_new) < abs(best_f):
                    best_x, best_f = x_new, f_new
                if abs(f_new) <= 1e-8:
                    return x_new
                x_prev, f_prev = x_curr, f_curr
                x_curr, f_curr = x_new, f_new
                continue

            x_new = x_curr - f_curr * (x_curr - x_prev) / denom

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

            if abs(f_new) < abs(best_f):
                best_x, best_f = x_new, f_new

            if abs(f_new) <= 1e-8:
                return x_new

            # Update: keep two most recent
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_new, f_new

        # Final polish with secant at tighter tolerance
        if abs(best_f) > 1e-8 and abs(best_f) < 1.0:
            x_curr = best_x
            f_curr = best_f
            h = 1e-8 * max(1.0, abs(x_curr))
            try:
                f_xh = f(x_curr + h)
            except Exception:
                f_xh = None

            if f_xh is not None and math.isfinite(f_xh):
                x_prev = x_curr + h
                f_prev = f_xh
                for it in range(30):
                    if not elapsed_ok():
                        break
                    if abs(f_curr) <= 1e-8:
                        return x_curr

                    denom = f_curr - f_prev
                    if abs(denom) < 1e-14:
                        break

                    x_new = x_curr - f_curr * (x_curr - x_prev) / denom

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
                        if abs(f_new) < abs(best_f):
                            best_x, best_f = x_new, f_new
                        x_prev, f_prev = x_curr, f_curr
                        x_curr, f_curr = x_new, f_new
                    else:
                        break

    return best_x
