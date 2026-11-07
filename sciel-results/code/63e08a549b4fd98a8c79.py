"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """Find a root of f near x0. Returns x with |f(x)| <= 1e-8."""
    t_start = time.monotonic()
    DEADLINE = 145.0  # seconds, safety margin below 150s kill

    x = float(x0)

    # --- Quick check at start ---
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x

    # --- Phase 1: Try to establish a bracket near x0 ---
    # Expand interval around x0 to find sign change
    a, b = x, x
    fa, fb = fx, fx
    step = 1.0
    bracketed = False
    best_x, best_f = x, fx

    for i in range(40):
        if time.monotonic() - t_start > DEADLINE:
            break
        a = x - step
        b = x + step
        fa = f(a)
        fb = f(b)
        if not (math.isfinite(fa) and math.isfinite(fb)):
            step *= 2.0
            continue
        # Track best
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

    # --- Phase 2: Safeguarded root finding ---
    # Use a combination of bisection and secant/Newton for fast convergence
    if bracketed:
        # Safeguarded secant method with bisection fallback
        lo, hi = a, b
        flo, fhi = fa, fb
        # Ensure flo * fhi < 0
        # Use secant from the two bracket endpoints
        x_prev = hi
        f_prev = fhi
        x_curr = lo
        f_curr = flo

        for it in range(80):
            if time.monotonic() - t_start > DEADLINE:
                break
            if abs(f_curr) <= 1e-8:
                return x_curr
            if abs(f_prev) <= 1e-8:
                return x_prev

            # Secant step
            if f_prev != f_curr:
                x_sec = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            else:
                x_sec = 0.5 * (x_curr + x_prev)

            # Check if secant step stays within bracket
            if (x_sec > lo and x_sec < hi):
                x_new = x_sec
            else:
                # Bisection step
                x_new = 0.5 * (lo + hi)

            f_new = f(x_new)
            if not math.isfinite(f_new):
                # Fall back to bisection
                x_new = 0.5 * (lo + hi)
                f_new = f(x_new)

            if abs(f_new) <= 1e-8:
                return x_new

            # Update bracket
            if f_new * f_curr < 0.0:
                lo, hi = x_curr, x_new
                flo, fhi = f_curr, f_new
            else:
                lo, hi = x_new, x_prev
                flo, fhi = f_new, f_prev

            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_new, f_new

        # Return best of the bracket
        if abs(f_curr) <= abs(f_prev):
            return x_curr
        else:
            return x_prev

    else:
        # No bracket found; use Newton's method with finite-difference derivative
        # and safeguarding
        x = x
        fx = best_f if abs(best_f) < abs(fx) else fx
        if abs(fx) <= 1e-8:
            return x

        best_x, best_f = x, fx

        for it in range(60):
            if time.monotonic() - t_start > DEADLINE:
                break
            if abs(fx) <= 1e-8:
                return x

            # Finite difference derivative
            h = 1e-7 * max(1.0, abs(x))
            f_xh = f(x + h)
            if not math.isfinite(f_xh):
                h = -h
                f_xh = f(x + h)
            if not math.isfinite(f_xh):
                break

            d = (f_xh - fx) / h
            if d == 0 or not math.isfinite(d):
                break

            x_new = x - fx / d

            # Safeguard: limit step size
            max_step = max(1.0, abs(x)) * 10.0
            if abs(x_new - x) > max_step:
                x_new = x + max_step * (1.0 if x_new > x else -1.0)

            fx_new = f(x_new)
            if not math.isfinite(fx_new):
                # Try smaller step
                x_new = x - 0.5 * fx / d
                fx_new = f(x_new)
                if not math.isfinite(fx_new):
                    break

            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
                if abs(fx) < abs(best_f):
                    best_x, best_f = x, fx
            else:
                # Damped step
                x_new = 0.5 * (x + x_new)
                fx_new = f(x_new)
                if math.isfinite(fx_new) and abs(fx_new) < abs(fx):
                    x, fx = x_new, fx_new
                    if abs(fx) < abs(best_f):
                        best_x, best_f = x, fx
                else:
                    break

        # Return best found
        if abs(best_f) < abs(fx):
            return best_x
        return x

    # --- Final polish: if we have a candidate, try to tighten it ---
    # (The above loops already return early on success, so this is rarely reached)
    return x
# EVOLVE-BLOCK-END
