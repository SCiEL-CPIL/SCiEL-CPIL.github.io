"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
import time as _time

def find_root(f, x0):
    """
    Find a root of f near x0 using a multi-strategy approach:
    1. Try secant locally first (fast for oscillatory fns near a root)
    2. If that fails, bracket and use Brent's method
    3. Polish with secant for tighter convergence
    4. Deadline guard to stay under 150s
    """
    start_time = _time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x

    # Track best solution found so far
    best_x = x
    best_fx = fx
    best_abs = abs(fx)

    def _check_deadline():
        return (_time.time() - start_time) > DEADLINE

    # Strategy 1: Local secant with a meaningful initial step
    # Use a step that is large enough to get a good derivative estimate
    # but small enough to likely be in the same oscillation period
    x_prev = x
    fx_prev = fx
    # Try a moderate initial step; for oscillatory fns, 0.1 is a reasonable guess
    # that captures local behavior without jumping to a different period
    step0 = 0.1 * max(1.0, abs(x))
    x_curr = x + step0
    fx_curr = f(x_curr)

    if abs(fx_curr) <= 1e-8:
        return x_curr

    # Also try negative direction as second initial point
    x_curr_neg = x - step0
    fx_curr_neg = f(x_curr_neg)

    if abs(fx_curr_neg) <= 1e-8:
        return x_curr_neg

    # Pick the better of the two initial points for secant
    if abs(fx_curr_neg) < abs(fx_curr):
        x_curr, fx_curr = x_curr_neg, fx_curr_neg

    if abs(fx_curr) < best_abs:
        best_x = x_curr
        best_fx = fx_curr
        best_abs = abs(fx_curr)

    for _ in range(25):
        if _check_deadline():
            break
        if abs(fx_curr) < best_abs:
            best_x = x_curr
            best_fx = fx_curr
            best_abs = abs(fx_curr)

        # Secant step
        denom = fx_curr - fx_prev
        if denom == 0 or not math.isfinite(denom):
            break
        x_new = x_curr - fx_curr * (x_curr - x_prev) / denom

        if not math.isfinite(x_new):
            break

        x_prev, fx_prev = x_curr, fx_curr
        x_curr = x_new
        fx_curr = f(x_curr)

        if abs(fx_curr) <= 1e-8:
            return x_curr

    if abs(fx_curr) < best_abs:
        best_x = x_curr
        best_fx = fx_curr
        best_abs = abs(fx_curr)

    if _check_deadline():
        return best_x

    # Strategy 2: Bracket expansion and Brent's method
    a = x0
    fa = f(a)

    if abs(fa) <= 1e-8:
        return a

    # Try expanding in both directions, starting with small steps
    h = 1e-3
    max_h = 1e6
    bracket_found = False
    b = None
    fb = None

    # Expand outward, checking both directions at each h
    while h <= max_h:
        if _check_deadline():
            break

        # Try positive direction
        x_pos = a + h
        fx_pos = f(x_pos)
        if abs(fx_pos) <= 1e-8:
            return x_pos
        if fa * fx_pos < 0:
            b = x_pos
            fb = fx_pos
            bracket_found = True
            break

        # Try negative direction
        x_neg = a - h
        fx_neg = f(x_neg)
        if abs(fx_neg) <= 1e-8:
            return x_neg
        if fa * fx_neg < 0:
            b = x_neg
            fb = fx_neg
            bracket_found = True
            break

        h *= 2.0

    if not bracket_found:
        # Try a few well-spaced probe points for wider search
        # These are geometrically spaced to cover a wide range efficiently
        probe_points = [1e-2, 5e-2, 0.25, 1.0, 5.0, 25.0, 100.0, 500.0, 2000.0, 10000.0]
        for pp in probe_points:
            if _check_deadline():
                break
            for sign in (1.0, -1.0):
                xp = a + sign * pp
                fxp = f(xp)
                if abs(fxp) <= 1e-8:
                    return xp
                if fa * fxp < 0:
                    b = xp
                    fb = fxp
                    bracket_found = True
                    break
            if bracket_found:
                break

    if not bracket_found:
        # Last resort: try to use the best point found
        return best_x

    # Use Brent's method (implement inline to avoid scipy overhead and have full control)
    def _brentq_impl(f_func, a_val, b_val, tol=1e-14, maxiter=100):
        fa_val = f_func(a_val)
        fb_val = f_func(b_val)
        if fa_val == 0.0:
            return a_val
        if fb_val == 0.0:
            return b_val
        if fa_val * fb_val > 0:
            raise ValueError("f(a) and f(b) must have different signs")

        c = a_val
        fc = fa_val
        d = b_val - a_val
        e = d

        for _ in range(maxiter):
            if _check_deadline():
                break
            if fb_val == 0.0:
                return b_val
            if fa_val == 0.0:
                return a_val
            if fa_val * fb_val > 0:
                a_val = b_val
                fa_val = fb_val
                d = e = b_val - a_val
            elif abs(fc) < abs(fb_val):
                a_val = b_val
                fa_val = fb_val
                b_val = c
                fb_val = fc
                c = d
                fc = fa_val
                d = e = b_val - a_val

            # Inverse quadratic interpolation
            tol1 = 2.220446049250313e-16 * max(1.0, abs(b_val))
            m = 0.5 * (a_val - b_val)
            if abs(m) <= tol1 or fb_val == 0.0:
                return b_val

            if abs(e) >= tol1 and abs(fa_val) > abs(fb_val):
                s = fb_val / fa_val
                if a_val == c:
                    p = 2.0 * m * s
                    q = 1.0 - s
                else:
                    q = fa_val / fc
                    r = fb_val / fc
                    p = s * (2.0 * m * q * (q - r) - (b_val - c) * (r - 1.0))
                    q = (q - 1.0) * (r - 1.0) * (s - 1.0)
                if p > 0.0:
                    q = -q
                elif p < 0.0:
                    p = -p
                if 2.0 * p < min(3.0 * m * q - abs(tol1 * q), abs(e * q)):
                    e = d
                    d = p / q
                else:
                    d = m
                    e = d
            else:
                d = m
                e = d

            if abs(d) < tol1:
                d = tol1 if d > 0 else -tol1

            c = b_val
            fc = fb_val
            b_val = b_val + d
            fb_val = f_func(b_val)

        return b_val

    try:
        root = _brentq_impl(f, a, b)
        if math.isfinite(root):
            # Polish pass: a few secant iterations to tighten |f(x)|
            x_polish = root
            fx_polish = f(root)
            if abs(fx_polish) <= 1e-8:
                return root
            # Use the bracket endpoints for secant polish
            x_p1 = a
            fx_p1 = fa
            x_p2 = root
            fx_p2 = fx_polish
            for _ in range(10):
                if _check_deadline():
                    break
                denom = fx_p2 - fx_p1
                if denom == 0 or not math.isfinite(denom):
                    break
                x_new = x_p2 - fx_p2 * (x_p2 - x_p1) / denom
                if not math.isfinite(x_new):
                    break
                fx_new = f(x_new)
                if abs(fx_new) <= 1e-8:
                    return x_new
                x_p1, fx_p1 = x_p2, fx_p2
                x_p2, fx_p2 = x_new, fx_new
            return x_polish
    except Exception:
        pass

    # Fallback: return best point found
    return best_x
# EVOLVE-BLOCK-END
