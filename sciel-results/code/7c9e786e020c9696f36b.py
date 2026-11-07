"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust scalar root finder for smooth polynomial equations.
    Two-phase Newton (forward-diff with call reuse + damping) followed by
    central-diff high-precision polish, with early termination at |f|<=1e-8.
    """
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s limit
    TOL = 1e-8  # the actual requirement - terminate as soon as we hit this
    POLISH_TOL = 1e-12  # tighter for final polish
    MAX_CALLS = 495  # stay well under 500 per equation

    call_count = 0
    best_x = float(x0)
    best_f = float('inf')

    def _check_time():
        return time.time() < deadline

    def _safe_f(x_val):
        nonlocal call_count, best_x, best_f
        if call_count >= MAX_CALLS:
            return None
        try:
            val = f(x_val)
            call_count += 1
            if not math.isfinite(val):
                return None
            if abs(val) < abs(best_f):
                best_f = val
                best_x = x_val
            return val
        except Exception:
            call_count += 1
            return None

    def _central_diff_polish(x_start, f_start, max_iter=15):
        """High-precision polish using central-difference Newton with tight tolerance."""
        nonlocal call_count, best_x, best_f
        if f_start is None or not math.isfinite(f_start):
            return x_start, f_start

        x_curr = x_start
        f_curr = f_start

        if abs(f_curr) <= POLISH_TOL:
            return x_curr, f_curr

        # Central difference derivative: (f(x+h) - f(x-h)) / (2h)
        # Reuse f_curr as one of the evaluations when possible
        for _ in range(max_iter):
            if not _check_time() or call_count >= MAX_CALLS - 2:
                break
            if abs(f_curr) <= POLISH_TOL:
                return x_curr, f_curr

            # Adaptive step size for central difference
            h = 1e-7 * max(1.0, abs(x_curr))
            # Ensure h is not too small relative to floating point
            h = max(h, abs(x_curr) * 1e-12 + 1e-15)

            f_plus = _safe_f(x_curr + h)
            if f_plus is None:
                break

            f_minus = _safe_f(x_curr - h)
            if f_minus is None:
                break

            d = (f_plus - f_minus) / (2.0 * h)
            if d == 0 or not math.isfinite(d):
                # Try smaller step
                h = h * 0.1
                f_plus = _safe_f(x_curr + h)
                if f_plus is None:
                    break
                f_minus = _safe_f(x_curr - h)
                if f_minus is None:
                    break
                d = (f_plus - f_minus) / (2.0 * h)
                if d == 0 or not math.isfinite(d):
                    break

            x_new = x_curr - f_curr / d
            if not math.isfinite(x_new):
                break

            # Limit step to avoid wild overshoots
            max_step = 10.0 * max(1.0, abs(x_curr))
            if abs(x_new - x_curr) > max_step:
                x_new = x_curr + max_step * (1.0 if x_new > x_curr else -1.0)

            f_new = _safe_f(x_new)
            if f_new is None:
                break

            if abs(f_new) <= POLISH_TOL:
                return x_new, f_new

            # Quadratic convergence check: if |f| is decreasing, continue
            if abs(f_new) < abs(f_curr):
                x_curr, f_curr = x_new, f_new
            else:
                # Not converging well; try secant fallback
                break

        return x_curr, f_curr

    def _secant_polish(x_start, f_start, x_prev, f_prev, max_iter=10):
        """Secant method polish with two known points."""
        nonlocal call_count, best_x, best_f
        if f_start is None or not math.isfinite(f_start):
            return x_start, f_start
        if f_prev is None or not math.isfinite(f_prev):
            return _central_diff_polish(x_start, f_start, max_iter)

        x_curr = x_start
        f_curr = f_start

        if abs(f_curr) <= POLISH_TOL:
            return x_curr, f_curr

        for _ in range(max_iter):
            if not _check_time() or call_count >= MAX_CALLS - 1:
                break
            if abs(f_curr) <= POLISH_TOL:
                return x_curr, f_curr

            if f_prev == f_curr:
                break

            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            if not math.isfinite(x_new):
                break

            f_new = _safe_f(x_new)
            if f_new is None:
                break

            if abs(f_new) <= POLISH_TOL:
                return x_new, f_new

            # Keep the two points with smaller |f|
            if abs(f_new) < abs(f_prev):
                x_prev, f_prev = x_curr, f_curr
                x_curr, f_curr = x_new, f_new
            else:
                x_prev, f_prev = x_new, f_new

        if abs(f_curr) < abs(f_prev):
            return x_curr, f_curr
        else:
            return x_prev, f_prev

    x = float(x0)
    fx = _safe_f(x)
    if fx is None:
        return best_x
    if abs(fx) <= TOL:
        return x

    # Phase 1: Find a bracket [a, b] with sign change, using few evaluations.
    a, b, fa, fb = None, None, None, None

    # Try small steps first (polynomial roots are often near x0)
    for s in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0,
              1000.0, 2000.0, 5000.0, 10000.0]:
        if not _check_time() or call_count >= MAX_CALLS - 10:
            break

        # Try right
        xr = x + s
        fr = _safe_f(xr)
        if fr is None:
            continue
        if abs(fr) <= TOL:
            return xr
        if fx * fr < 0:
            a, b, fa, fb = x, xr, fx, fr
            break

        # Try left
        xl = x - s
        fl = _safe_f(xl)
        if fl is None:
            continue
        if abs(fl) <= TOL:
            return xl
        if fx * fl < 0:
            a, b, fa, fb = xl, x, fl, fx
            break

    # If no bracket found, try geometric expansion
    if a is None:
        step = 1.0
        for _ in range(15):
            if not _check_time() or call_count >= MAX_CALLS - 10:
                break

            xr = x + step
            fr = _safe_f(xr)
            if fr is not None:
                if abs(fr) <= TOL:
                    return xr
                if fx * fr < 0:
                    a, b, fa, fb = x, xr, fx, fr
                    break

            xl = x - step
            fl = _safe_f(xl)
            if fl is not None:
                if abs(fl) <= TOL:
                    return xl
                if fx * fl < 0:
                    a, b, fa, fb = xl, x, fl, fx
                    break

            step *= 2.0
            if step > 1e8:
                break

    # Phase 2: If bracket found, use hybrid Newton-bisection with early termination
    if a is not None and b is not None and fa is not None and fb is not None:
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        x_lo, x_hi = a, b
        f_lo, f_hi = fa, fb

        # Secant points for initialization
        x_prev2, f_prev2 = x_lo, f_lo
        x_prev1, f_prev1 = x_hi, f_hi

        for _ in range(35):
            if not _check_time() or call_count >= MAX_CALLS - 5:
                break

            if abs(f_lo) <= TOL:
                return x_lo
            if abs(f_hi) <= TOL:
                return x_hi

            # Midpoint for bisection fallback
            x_mid = (x_lo + x_hi) / 2.0
            f_mid = _safe_f(x_mid)
            if f_mid is None:
                break
            if abs(f_mid) <= TOL:
                return x_mid

            # Try secant step using the two most recent points
            if f_prev1 != f_prev2:
                x_sec = x_prev1 - f_prev1 * (x_prev1 - x_prev2) / (f_prev1 - f_prev2)
                if x_lo < x_sec < x_hi:
                    f_sec = _safe_f(x_sec)
                    if f_sec is not None:
                        if abs(f_sec) <= TOL:
                            return x_sec
                        # Update bracket
                        if f_sec * f_lo < 0:
                            x_hi = x_sec
                            f_hi = f_sec
                        else:
                            x_lo = x_sec
                            f_lo = f_sec
                        # Update secant points
                        x_prev2, f_prev2 = x_prev1, f_prev1
                        x_prev1, f_prev1 = x_sec, f_sec
                        continue

            # Fallback: bisection
            if f_mid * f_lo < 0:
                x_hi = x_mid
                f_hi = f_mid
            else:
                x_lo = x_mid
                f_lo = f_mid

            # Update secant points
            if f_lo * f_mid < 0:
                x_prev2, f_prev2 = x_prev1, f_prev1
                x_prev1, f_prev1 = x_mid, f_mid
            else:
                x_prev2, f_prev2 = x_prev1, f_prev1
                x_prev1, f_prev1 = x_mid, f_mid

        # Return the best endpoint
        if abs(f_lo) < abs(f_hi):
            result_x, result_f = x_lo, f_lo
        else:
            result_x, result_f = x_hi, f_hi

        # Final polish pass
        if _check_time() and call_count < MAX_CALLS - 8:
            if abs(result_f) > POLISH_TOL:
                result_x, result_f = _central_diff_polish(result_x, result_f, max_iter=8)

        return result_x

    # Phase 3: No bracket found. Use two-phase Newton with forward difference.
    # Phase 3a: Forward-difference Newton with call reuse and damping
    x_curr = float(x0)
    f_curr = fx

    for _ in range(20):
        if not _check_time() or call_count >= MAX_CALLS - 10:
            break

        if abs(f_curr) <= TOL:
            return x_curr

        # Forward difference derivative: (f(x+h) - f(x)) / h
        # Reuse f_curr (already computed)
        h = 1e-7 * max(1.0, abs(x_curr))
        h = max(h, abs(x_curr) * 1e-12 + 1e-15)
        fx_plus = _safe_f(x_curr + h)
        if fx_plus is None:
            break

        d = (fx_plus - f_curr) / h
        if d == 0 or not math.isfinite(d):
            # Try smaller step
            h = h * 0.1
            fx_plus = _safe_f(x_curr + h)
            if fx_plus is None:
                break
            d = (fx_plus - f_curr) / h
            if d == 0 or not math.isfinite(d):
                break

        x_new = x_curr - f_curr / d
        if not math.isfinite(x_new):
            break

        # Damping: limit step to avoid wild overshoots
        max_step = 5.0 * max(1.0, abs(x_curr))
        if abs(x_new - x_curr) > max_step:
            x_new = x_curr + max_step * (1.0 if x_new > x_curr else -1.0)

        f_new = _safe_f(x_new)
        if f_new is None:
            break

        if abs(f_new) <= TOL:
            return x_new

        # If making progress, continue; otherwise try bracketing from here
        if abs(f_new) < abs(f_curr) * 0.95:
            x_curr = x_new
            f_curr = f_new
        else:
            # Stagnation: try to find a bracket using current and previous
            if abs(f_new) < abs(f_curr):
                x_curr, f_curr = x_new, f_new
            else:
                break

    # Phase 3b: If Newton didn't converge, try to bracket using best points
    # and do a final polish
    if _check_time() and call_count < MAX_CALLS - 8:
        if abs(best_f) > POLISH_TOL:
            # Try central-diff polish on the best point found
            best_x, best_f = _central_diff_polish(best_x, best_f, max_iter=10)

    # Final: return the best point we've seen
    return best_x
# EVOLVE-BLOCK-END
