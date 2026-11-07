import math
import time

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 490
    deadline = time.time() + 145.0

    calls = 0
    best_x = float(x0)
    best_abs = float('inf')

    def safe_f(x):
        nonlocal calls, best_x, best_abs
        if calls >= MAX_CALLS or time.time() > deadline:
            return float('inf')
        calls += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            a = abs(val)
            if a < best_abs:
                best_abs = a
                best_x = x
            return val
        except Exception:
            return float('inf')

    x0 = float(x0)

    # Evaluate at x0
    fx0 = safe_f(x0)
    if abs(fx0) <= TOL:
        return x0
    if not math.isfinite(fx0):
        # x0 is in a bad region, try nearby
        for dx in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0]:
            x_try = x0 + dx
            fx_try = safe_f(x_try)
            if abs(fx_try) <= TOL:
                return x_try
            if math.isfinite(fx_try):
                best_x = x_try
                best_abs = abs(fx_try)
                break
        return best_x

    # Phase 1: Multi-scale scanning to find a bracket
    # Try progressively larger windows, checking multiple points within each
    bracket_a = None
    bracket_b = None
    fa_val = None
    fb_val = None

    scales = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    for scale in scales:
        if calls >= MAX_CALLS - 50 or time.time() > deadline:
            break

        # Check sign changes at multiple points within the window
        # Use a denser grid for small scales, sparser for large
        if scale <= 0.5:
            n_pts = 11
        elif scale <= 5.0:
            n_pts = 21
        else:
            n_pts = 31

        prev_x = None
        prev_f = None

        for i in range(n_pts):
            if calls >= MAX_CALLS - 50 or time.time() > deadline:
                break
            t = -1.0 + 2.0 * i / (n_pts - 1)
            x_pt = x0 + t * scale
            f_pt = safe_f(x_pt)

            if not math.isfinite(f_pt):
                prev_x = None
                prev_f = None
                continue

            if prev_f is not None and math.isfinite(prev_f):
                if prev_f * f_pt < 0:
                    bracket_a = prev_x
                    bracket_b = x_pt
                    fa_val = prev_f
                    fb_val = f_pt
                    break
                # Also check if either point is a root
                if abs(f_pt) <= TOL:
                    return x_pt
                if abs(prev_f) <= TOL:
                    return prev_x

            prev_x = x_pt
            prev_f = f_pt

        if bracket_a is not None:
            break

    if bracket_a is not None:
        # Phase 2: Refine using Brent's method (via scipy if available, else manual)
        a, b = bracket_a, bracket_b
        fa = fa_val if fa_val is not None else safe_f(a)
        fb = fb_val if fb_val is not None else safe_f(b)

        # Ensure fa and fb have opposite signs
        if fa * fb > 0:
            # Re-evaluate to be safe
            fa = safe_f(a)
            fb = safe_f(b)
            if fa * fb > 0:
                # No valid bracket, return best
                return best_x

        # Use a robust combination of bisection and secant (Brent-like)
        # Track previous points for secant
        x_mid = (a + b) / 2.0
        f_mid = safe_f(x_mid)

        if abs(f_mid) <= TOL:
            return x_mid

        # Determine which half contains the sign change
        if fa * f_mid < 0:
            b = x_mid
            fb = f_mid
        else:
            a = x_mid
            fa = f_mid

        # Now we have a, b with fa*fb < 0
        # Use a hybrid: try secant, fall back to bisection
        x_prev = a
        f_prev = fa
        x_curr = b
        f_curr = fb

        for _ in range(60):
            if calls >= MAX_CALLS - 5 or time.time() > deadline:
                break

            if abs(f_curr) <= TOL:
                return x_curr
            if abs(f_prev) <= TOL:
                return x_prev

            if abs(b - a) < 1e-15:
                break

            # Try secant step
            if f_prev != f_curr:
                x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            else:
                x_new = (a + b) / 2.0

            # Ensure x_new is within the bracket
            if not (a < x_new < b):
                x_new = (a + b) / 2.0

            f_new = safe_f(x_new)

            if not math.isfinite(f_new):
                x_new = (a + b) / 2.0
                f_new = safe_f(x_new)
                if not math.isfinite(f_new):
                    break

            if abs(f_new) <= TOL:
                return x_new

            # Update bracket
            if fa * f_new < 0:
                b = x_new
                fb = f_new
                x_curr = x_new
                f_curr = f_new
            else:
                a = x_new
                fa = f_new
                x_curr = x_new
                f_curr = f_new

            x_prev = x_curr
            f_prev = f_curr

        # Return the point with smaller |f|
        if abs(fa) < abs(fb):
            return a
        else:
            return b

    # Phase 3: No bracket found; use secant method with damping
    x_prev = x0 - 0.1
    fx_prev = safe_f(x_prev)
    x_curr = x0
    fx_curr = fx0

    if abs(fx_curr) <= TOL:
        return x_curr

    for _ in range(80):
        if calls >= MAX_CALLS - 5 or time.time() > deadline:
            break

        if abs(fx_prev) < abs(fx_curr):
            x_prev, x_curr = x_curr, x_prev
            fx_prev, fx_curr = fx_curr, fx_prev

        if abs(fx_curr) <= TOL:
            return x_curr

        if fx_prev == fx_curr:
            break

        x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)

        # Damp if step is too large
        if abs(x_new - x_curr) > 10.0 * max(1.0, abs(x_curr)):
            x_new = x_curr - 10.0 * max(1.0, abs(x_curr)) * (1.0 if fx_curr > 0 else -1.0)

        fx_new = safe_f(x_new)

        if abs(fx_new) <= TOL:
            return x_new

        if not math.isfinite(fx_new):
            x_new = x_curr - 0.1 * (x_curr - x_prev)
            fx_new = safe_f(x_new)
            if not math.isfinite(fx_new):
                break

        x_prev, fx_prev = x_curr, fx_curr
        x_curr, fx_curr = x_new, fx_new

    return best_x
