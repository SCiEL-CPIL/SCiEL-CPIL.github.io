import math
import time

# EVOLVE-BLOCK-START
def _is_finite(x):
    return math.isfinite(x)

def find_root(f, x0):
    deadline = time.monotonic() + 110.0  # safety margin before 120s kill

    x0 = float(x0)
    if not _is_finite(x0):
        x0 = 0.0

    # --- Phase 1: evaluate at x0 ---
    fx0 = f(x0)
    if not _is_finite(fx0):
        fx0 = 0.0
    if abs(fx0) <= 1e-8:
        return x0

    best_x = x0
    best_abs = abs(fx0)

    # --- Phase 2: build a bracket via geometric expansion ---
    # Try to find x_lo, x_hi with f of opposite signs.
    x_lo = x0
    x_hi = x0
    f_lo = fx0
    f_hi = fx0

    scale = 1.0
    bracket_found = False
    # Track previous point for secant fallback
    prev_x = None
    prev_f = None

    for i in range(60):
        if time.monotonic() > deadline:
            break
        d = scale
        if i % 2 == 0:
            x_try = x0 + d
        else:
            x_try = x0 - d

        if not _is_finite(x_try):
            scale *= 2.0
            continue

        f_try = f(x_try)
        if not _is_finite(f_try):
            scale *= 2.0
            continue

        if abs(f_try) < best_abs:
            best_abs = abs(f_try)
            best_x = x_try

        if abs(f_try) <= 1e-8:
            return x_try

        # Check for sign change with x0
        if f_try * fx0 < 0:
            if x_try > x0:
                x_lo, x_hi = x0, x_try
                f_lo, f_hi = fx0, f_try
            else:
                x_lo, x_hi = x_try, x0
                f_lo, f_hi = f_try, fx0
            bracket_found = True
            break

        # Also check sign change with previous point
        if prev_x is not None and f_try * prev_f < 0:
            if x_try > prev_x:
                x_lo, x_hi = prev_x, x_try
                f_lo, f_hi = prev_f, f_try
            else:
                x_lo, x_hi = x_try, prev_x
                f_lo, f_hi = f_try, prev_f
            bracket_found = True
            break

        prev_x = x_try
        prev_f = f_try

        if i % 2 == 1:
            scale *= 2.0

    if not bracket_found:
        # No bracket found. Use secant method from best points.
        # We have at least x0 and possibly prev_x.
        if prev_x is not None and prev_x != x0:
            x_a = x0
            f_a = fx0
            x_b = prev_x
            f_b = prev_f
        else:
            # Try a few more points to get a second one
            x_b = x0 + 1.0
            f_b = f(x_b)
            if not _is_finite(f_b):
                f_b = 0.0
            x_a = x0
            f_a = fx0

        for _ in range(80):
            if time.monotonic() > deadline:
                break
            if abs(f_b) < abs(f_a):
                x_a, f_a = x_b, f_b
            if abs(f_a) <= 1e-8:
                return x_a
            if not _is_finite(x_a) or not _is_finite(f_a):
                break
            if f_b == f_a:
                x_new = x_a + 0.1 * max(1.0, abs(x_a))
            else:
                x_new = x_a - f_a * (x_a - x_b) / (f_a - f_b)
            if not _is_finite(x_new):
                x_new = x_a + 0.1 * max(1.0, abs(x_a))
            f_new = f(x_new)
            if not _is_finite(f_new):
                f_new = 0.0
            x_b = x_new
            f_b = f_new
        return x_a if abs(f_a) <= abs(f_b) else x_b

    # --- Phase 3: safeguarded secant with bisection fallback ---
    # We have [x_lo, x_hi] with f_lo * f_hi < 0
    x_prev = x_lo
    f_prev = f_lo
    x_curr = x_hi
    f_curr = f_hi

    for _ in range(200):
        if time.monotonic() > deadline:
            break
        if abs(f_curr) <= 1e-8:
            return x_curr
        if abs(f_prev) <= 1e-8:
            return x_prev

        # Secant prediction
        if f_curr != f_prev:
            x_sec = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        else:
            x_sec = 0.5 * (x_lo + x_hi)

        if not _is_finite(x_sec):
            x_sec = 0.5 * (x_lo + x_hi)

        # Safeguard: if secant step is outside bracket, use bisection
        if x_sec <= x_lo or x_sec >= x_hi:
            x_new = 0.5 * (x_lo + x_hi)
        else:
            x_new = x_sec

        f_new = f(x_new)
        if not _is_finite(f_new):
            x_new = 0.5 * (x_lo + x_hi)
            f_new = f(x_new)
            if not _is_finite(f_new):
                break

        # Update bracket
        if f_new * f_lo <= 0:
            x_hi = x_new
            f_hi = f_new
        else:
            x_lo = x_new
            f_lo = f_new

        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_new
        f_curr = f_new

        # Convergence check on interval
        if abs(x_hi - x_lo) < 1e-14 * max(1.0, abs(x_lo), abs(x_hi)):
            break

    # --- Phase 4: high-precision polish (Newton) ---
    x = x_curr if abs(f_curr) <= abs(f_prev) else x_prev
    fx = f_curr if abs(f_curr) <= abs(f_prev) else f_prev

    for _ in range(10):
        if time.monotonic() > deadline:
            break
        if abs(fx) <= 1e-12:
            return x
        h = 1e-8 * max(1.0, abs(x))
        f_plus = f(x + h)
        f_minus = f(x - h)
        if not _is_finite(f_plus) or not _is_finite(f_minus):
            break
        d = (f_plus - f_minus) / (2.0 * h)
        if d == 0 or not _is_finite(d):
            break
        x_new = x - fx / d
        if not _is_finite(x_new):
            break
        fx_new = f(x_new)
        if not _is_finite(fx_new):
            break
        x = x_new
        fx = fx_new

    if abs(fx) <= 1e-8:
        return x

    # Return best candidate
    return x
# EVOLVE-BLOCK-END
