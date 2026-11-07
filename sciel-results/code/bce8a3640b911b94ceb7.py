"""Root finder for smooth polynomial equations - secant-based hybrid."""
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
                    x, fx = xt, fxt
                    break
            if math.isfinite(fx):
                break
        if not math.isfinite(fx):
            return x0

    if abs(fx) <= 1e-8:
        return x

    # Track best
    best_x, best_f = x, fx

    # Phase 1: Try to find a bracket via expansion
    a, b = x, x
    fa, fb = fx, fx
    step = 1.0
    bracketed = False

    for i in range(30):
        if time.time() > deadline:
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
        # Brent-like method: combine bisection, secant, and inverse quadratic interpolation
        lo, hi = a, b
        flo, fhi = fa, fb
        # Ensure |flo| <= |fhi| so we keep the one with smaller magnitude as 'lo'
        if abs(flo) > abs(fhi):
            lo, hi = hi, lo
            flo, fhi = fhi, flo

        c = lo
        fc = flo
        d = lo - hi  # previous step
        e = d  # last interpolation step

        for it in range(100):
            if time.time() > deadline:
                break

            if abs(flo) <= 1e-8:
                return lo
            if abs(fhi) <= 1e-8:
                return hi

            # Convergence check
            if abs(hi - lo) < 1e-14:
                break

            # Inverse quadratic interpolation
            if (c != lo and c != hi):
                # Check if all three points are distinct
                # Compute the three terms
                t1 = flo * fhi
                t2 = flo * fc
                t3 = fhi * fc

                # Standard IQUO formula
                # x_new = hi - (hi - lo) * ( (hi-c)*t2 / ((lo-c)*t3) - (lo-c)*t1 / ((hi-c)*t2) )
                # But we need to be careful with the formula. Let's use the standard one.
                
                # Actually, let's use the standard Brent's method IQUO:
                # p = 2 * e * (t = flo / fhi)
                # If t != 1:
                #   q = 1 - t
                #   r = (hi - c) * (flo / fc)
                #   p = r * q - (hi - lo) * (t)
                #   e_old = e
                #   e = r * q - (hi - lo) * t
                #   if p > 0: e = -e
                #   p = p / (2 * abs(q))
                # else:
                #   p = e_old
                #   e = e_old
                
                t = flo / fhi
                if t != 1.0:
                    q = 1.0 - t
                    r = (hi - c) * (flo / fc) if fc != 0 else 0.0
                    p = r * q - (hi - lo) * t
                    e_old = e
                    e = r * q - (hi - lo) * t
                    if p > 0:
                        e = -e
                    p = p / (2.0 * abs(q))
                else:
                    p = e
                    e = e

                # Check if IQUO step is acceptable
                ok = True
                # p must be greater than half the smallest allowable step
                if not (p > 0.5 * min(2.0 * (hi - lo) * (abs(flo) - abs(fhi)) / abs(fhi), abs(hi - lo))):
                    ok = False
                # p must be less than the last step
                if not (p < abs(e)):
                    ok = False
                # p must be within the bracket
                if not ((hi - lo) * p < 0.5 * (hi - lo) * (abs(flo) - abs(fhi)) / abs(fhi)):
                    ok = False

                if ok:
                    d = e
                    e = p
                else:
                    # Bisection
                    e = 0.5 * (hi - lo)
                    d = e
            else:
                # Secant step
                if flo != fhi:
                    e = 0.5 * (hi - lo)
                    d = (hi - lo) * (flo / (flo - fhi))
                else:
                    e = 0.5 * (hi - lo)
                    d = e

            # If d is too small, use bisection
            if abs(d) < 1e-14:
                x_new = lo + d
            else:
                x_new = lo + d

            # Ensure x_new is within bracket
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

            # Update best
            if abs(f_new) < abs(best_f):
                best_x, best_f = x_new, f_new

            # Update bracket
            if f_new * flo < 0:
                hi, fhi = x_new, f_new
            else:
                c, fc = lo, flo
                lo, flo = x_new, f_new

            # Keep |flo| <= |fhi|
            if abs(flo) > abs(fhi):
                lo, hi = hi, lo
                flo, fhi = fhi, flo

        # Return best
        if abs(flo) <= abs(fhi):
            return lo
        else:
            return hi
    else:
        # No bracket: use safeguarded Newton with finite differences
        # Try a few Newton steps from x0 and nearby points
        candidates = [x]
        for dx in [1.0, -1.0, 0.1, -0.1, 0.01, -0.01, 10.0, -10.0]:
            candidates.append(x + dx)

        for x_start in candidates:
            if time.time() > deadline:
                break
            try:
                fx = f(x_start)
            except Exception:
                continue
            if not math.isfinite(fx):
                continue
            if abs(fx) <= 1e-8:
                return x_start
            if abs(fx) < abs(best_f):
                best_x, best_f = x_start, fx

            # Newton iterations
            x_curr = x_start
            f_curr = fx
            for it in range(50):
                if time.time() > deadline:
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

                x_new = x_curr - f_curr / d

                # Safeguard
                max_step = max(1.0, abs(x_curr)) * 10.0
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
                    # Damped step
                    x_new = 0.5 * (x_curr + x_new)
                    try:
                        f_new = f(x_new)
                    except Exception:
                        break
                    if math.isfinite(f_new) and abs(f_new) < abs(f_curr):
                        x_curr, f_curr = x_new, f_new
                        if abs(f_curr) < abs(best_f):
                            best_x, best_f = x_curr, f_curr
                    else:
                        break

    return best_x
# EVOLVE-BLOCK-END
