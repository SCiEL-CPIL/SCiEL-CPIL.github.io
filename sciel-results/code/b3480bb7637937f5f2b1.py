"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust scalar root finder using Regula Falsi with Brent-style safeguard.
    Uses inverse interpolation (linear) with bisection fallback, reusing
    bracket endpoint evaluations to minimize total function calls.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    TOL = 1e-9  # slightly stricter than 1e-8
    MAX_CALLS = 495  # stay well under 500

    x = float(x0)
    call_count = 0

    def _check_time():
        return time.time() < deadline

    def _safe_f(x_val):
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return None
        try:
            val = f(x_val)
            call_count += 1
            if not math.isfinite(val):
                return None
            return val
        except Exception:
            call_count += 1
            return None

    # Phase 1: Bracket finding via expanding search
    fx = _safe_f(x)
    if fx is None:
        return x
    if abs(fx) <= TOL:
        return x

    a, b, fa, fb = None, None, None, None
    step = 1.0

    for _ in range(35):
        if not _check_time() or call_count >= MAX_CALLS - 2:
            break

        # Try right
        xr = x + step
        fr = _safe_f(xr)
        if fr is not None:
            if abs(fr) <= TOL:
                return xr
            if fx * fr < 0:
                a, b, fa, fb = x, xr, fx, fr
                break

        # Try left
        xl = x - step
        fl = _safe_f(xl)
        if fl is not None:
            if abs(fl) <= TOL:
                return xl
            if fx * fl < 0:
                a, b, fa, fb = xl, x, fx, fl
                break

        step *= 2.0
        if step > 1e8:
            break

    # Phase 2: If bracket found, use Regula Falsi with Brent safeguard
    if a is not None and b is not None and fa is not None and fb is not None:
        # Ensure fa*fb < 0 (opposite signs)
        if fa * fb > 0:
            a, b, fa, fb = b, a, fb, fa

        # Track the two bracket endpoints; we always keep the one with smaller |f|
        # as the "slow" side and replace the other each iteration (Regula Falsi)
        # But with Brent's safeguard: if interpolation doesn't reduce bracket
        # sufficiently, fall back to bisection.

        # State: (x_lo, f_lo), (x_hi, f_hi) with f_lo * f_hi < 0
        x_lo, f_lo = a, fa
        x_hi, f_hi = b, fb

        # For Brent-style: track previous iteration's "new" point to detect
        # when interpolation is stalling
        x_prev_new = None
        f_prev_new = None
        stall_count = 0
        MIN_REDUCTION = 0.5  # bracket must shrink by at least this fraction

        for _ in range(80):
            if not _check_time() or call_count >= MAX_CALLS - 1:
                break

            if abs(f_lo) <= TOL:
                return x_lo
            if abs(f_hi) <= TOL:
                return x_hi

            bracket_width = abs(x_hi - x_lo)
            if bracket_width < 1e-15:
                # Return whichever endpoint is closer to zero
                return x_lo if abs(f_lo) <= abs(f_hi) else x_hi

            # Regula Falsi (inverse linear interpolation) step
            # x_new = x_lo - f_lo * (x_lo - x_hi) / (f_lo - f_hi)
            denom = f_lo - f_hi
            if abs(denom) < 1e-300:
                # Degenerate: just bisect
                x_new = (x_lo + x_hi) / 2.0
            else:
                x_new = x_lo - f_lo * (x_lo - x_hi) / denom

            # Brent's safeguard: ensure x_new is not too close to an endpoint
            # and that the bracket actually shrinks
            x_mid = (x_lo + x_hi) / 2.0

            # If interpolation point is too close to an endpoint or outside,
            # use bisection
            if (x_new <= x_lo or x_new >= x_hi or
                abs(x_new - x_lo) < 0.1 * bracket_width or
                abs(x_new - x_hi) < 0.1 * bracket_width):
                x_new = x_mid

            # Additional safeguard: if we've been stalling (interpolation
            # not reducing bracket much), force bisection
            if stall_count >= 2:
                x_new = x_mid
                stall_count = 0

            f_new = _safe_f(x_new)
            if f_new is None:
                break

            if abs(f_new) <= TOL:
                return x_new

            # Update bracket
            if f_new * f_lo < 0:
                x_hi, f_hi = x_new, f_new
            else:
                x_lo, f_lo = x_new, f_new

            # Check for stalling: if the new point is on the same side as
            # the previous new point and bracket hasn't shrunk much
            if x_prev_new is not None:
                new_width = abs(x_hi - x_lo)
                if new_width > MIN_REDUCTION * bracket_width:
                    stall_count += 1
                else:
                    stall_count = 0

            x_prev_new = x_new
            f_prev_new = f_new

        # Return best endpoint
        return x_lo if abs(f_lo) <= abs(f_hi) else x_hi

    # Phase 3: No bracket found — use secant/Newton hybrid
    # Try secant method from x0
    x_curr = x
    f_curr = fx  # already evaluated
    h = 1e-6 * max(1.0, abs(x_curr))
    x_prev = x_curr + h
    f_prev = _safe_f(x_prev)

    for _ in range(50):
        if not _check_time() or call_count >= MAX_CALLS - 1:
            break

        if f_prev is None:
            break
        if abs(f_prev) <= TOL:
            return x_prev

        denom = f_curr - f_prev
        if abs(denom) < 1e-300:
            # Try a different perturbation
            h *= 10.0
            x_prev = x_curr + h
            f_prev = _safe_f(x_prev)
            if f_prev is None:
                break
            denom = f_curr - f_prev
            if abs(denom) < 1e-300:
                break

        x_new = x_curr - f_curr * (x_curr - x_prev) / denom

        if not math.isfinite(x_new):
            break

        # Damping
        max_step = 10.0 * max(1.0, abs(x_curr))
        if abs(x_new - x_curr) > max_step:
            x_new = x_curr + max_step * (1 if x_new > x_curr else -1)

        f_new = _safe_f(x_new)
        if f_new is None:
            break

        if abs(f_new) <= TOL:
            return x_new

        # Update: keep the two most recent points
        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_new, f_new

    # Final: return best available
    if f_curr is not None and abs(f_curr) <= TOL:
        return x_curr
    if f_prev is not None and abs(f_prev) <= TOL:
        return x_prev
    return x_curr
# EVOLVE-BLOCK-END
