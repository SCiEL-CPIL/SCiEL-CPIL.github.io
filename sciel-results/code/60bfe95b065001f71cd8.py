"""Root finder for 3 fixed smooth polynomial equations; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def _is_finite(x):
    try:
        return x is not None and math.isfinite(x)
    except (TypeError, ValueError):
        return False

def _safe_f(f, x):
    try:
        y = f(x)
    except Exception:
        return None
    if not _is_finite(y):
        return None
    return float(y)

def find_root(f, x0):
    start = time.monotonic()
    deadline = start + 110.0
    MAX_CALLS = 400
    TOL = 1e-12
    calls = 0

    def g(x):
        nonlocal calls
        if calls >= MAX_CALLS:
            return None
        y = _safe_f(f, x)
        calls += 1
        return y

    def time_ok():
        return time.monotonic() < deadline

    best_x = None
    best_f = None
    best_abs = math.inf

    def consider(x, fx):
        nonlocal best_x, best_f, best_abs
        if fx is None or not _is_finite(x):
            return
        a = abs(fx)
        if a < best_abs:
            best_abs = a
            best_x = x
            best_f = fx

    # --- Phase 1: Evaluate at x0 and immediate neighbors ---
    x0 = float(x0)
    if not _is_finite(x0):
        x0 = 0.0

    fx0 = g(x0)
    consider(x0, fx0)
    if fx0 is not None and abs(fx0) <= TOL:
        return x0

    # Try a few nearby points to quickly find a bracket or a good starting point
    # This also helps if the root is very close to x0
    base_step = max(1.0, abs(x0)) * 0.5
    bracket_a = bracket_b = None
    bracket_fa = bracket_fb = None

    # Small steps first
    for mult in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0, 65536.0, 131072.0, 262144.0, 524288.0, 1048576.0, 2097152.0, 4194304.0, 8388608.0, 16777216.0, 33554432.0, 67108864.0, 134217728.0, 268435456.0, 536870912.0, 1073741824.0, 2147483648.0, 4294967296.0, 8589934592.0, 17179869184.0, 34359738368.0, 68719476736.0, 137438953472.0, 274877906944.0, 549755813888.0, 1099511627776.0, 2199023255552.0, 4398046511104.0, 8796093022208.0, 17592186044416.0, 35184372088832.0, 70368744177664.0, 140737488355328.0, 281474976710656.0, 562949953421312.0, 1125899906842624.0, 2251799813685248.0, 4503599627370496.0, 9007199254740992.0, 18014398509481984.0, 36028797018963968.0, 72057594037927936.0, 144115188075855872.0, 288230376151711744.0, 576460752303423488.0, 1152921504606846976.0, 2305843009213693952.0, 4611686018427387904.0, 9223372036854775808.0):
        if not time_ok():
            break
        d = base_step * mult
        for sign in (1.0, -1.0):
            trial = x0 + sign * d
            if not _is_finite(trial):
                continue
            ft = g(trial)
            consider(trial, ft)
            if ft is None:
                continue
            if abs(ft) <= TOL:
                return trial
            if fx0 is not None and fx0 * ft < 0.0:
                if fx0 < 0:
                    bracket_a, bracket_b = x0, trial
                    bracket_fa, bracket_fb = fx0, ft
                else:
                    bracket_a, bracket_b = trial, x0
                    bracket_fa, bracket_fb = ft, fx0
                break
        if bracket_a is not None:
            break

    # --- Phase 2: Main solver ---
    # Use a safeguarded secant method (Brent-like) if we have a bracket
    # Otherwise use pure secant with damping

    if bracket_a is not None and bracket_b is not None:
        a = bracket_a
        b = bracket_b
        fa = bracket_fa
        fb = bracket_fb

        # Ensure a < b for clarity, but keep track of f values
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Safeguarded secant / bisection hybrid
        # We maintain a bracket [a, b] where f(a)*f(b) < 0
        # And a secant prediction from the two most recent points
        
        # Initial secant points: a and b
        x_prev = a
        f_prev = fa
        x_curr = b
        f_curr = fb

        for _ in range(300):
            if not time_ok():
                break
            if calls >= MAX_CALLS:
                break
            if abs(f_curr) <= TOL:
                return x_curr
            if abs(f_prev) <= TOL:
                return x_prev

            # Secant step
            denom = f_curr - f_prev
            if denom == 0.0 or not _is_finite(denom):
                x_new = 0.5 * (a + b)
            else:
                x_new = x_curr - f_curr * (x_curr - x_prev) / denom

            if not _is_finite(x_new):
                x_new = 0.5 * (a + b)

            # Safeguard: if x_new is outside [a, b], use bisection
            if x_new <= a or x_new >= b:
                x_new = 0.5 * (a + b)

            f_new = g(x_new)
            consider(x_new, f_new)
            if f_new is None:
                break
            if abs(f_new) <= TOL:
                return x_new

            # Update bracket
            if f_new * fa <= 0:
                b = x_new
                fb = f_new
            else:
                a = x_new
                fa = f_new

            # Update secant points: shift x_prev to x_curr, x_curr to x_new
            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_new
            f_curr = f_new

            # Convergence check on interval width
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break

        # Return the best of the bracket endpoints
        if abs(fa) <= abs(fb):
            return a
        return b

    else:
        # No bracket found. Use secant method from x0 and a nearby point.
        # Pick the best point we found as the second starting point if it's different from x0.
        if best_x is not None and best_x != x0:
            xa = x0
            fa = fx0
            xb = best_x
            fb = best_f
        else:
            xa = x0
            fa = fx0
            # Try a small step
            h = max(1e-4, abs(xa) * 1e-4)
            xb = xa + h
            fb = g(xb)
            consider(xb, fb)
            if fb is None:
                # If we can't even evaluate a neighbor, return best
                if best_x is not None:
                    return best_x
                return x0

        if fa is None or fb is None:
            if best_x is not None:
                return best_x
            return x0

        if abs(fb) <= TOL:
            return xb
        if abs(fa) <= TOL:
            return xa

        # Secant method with damping
        ref_x = best_x if best_x is not None else xa
        ref_f = best_f if best_f is not None else fa

        for _ in range(300):
            if not time_ok():
                break
            if calls >= MAX_CALLS:
                break
            if abs(fb) <= TOL:
                return xb
            if abs(fa) <= TOL:
                return xa

            denom = fb - fa
            if denom == 0.0 or not _is_finite(denom):
                s = (xa + xb) / 2.0
            else:
                s = xb - fb * (xb - xa) / denom

            if not _is_finite(s):
                s = (xa + xb) / 2.0

            # Damping: don't go too far from the best point
            max_step = max(1.0, abs(ref_x)) * 1e6
            if abs(s - ref_x) > max_step:
                s = ref_x + math.copysign(max_step, s - ref_x)

            fs = g(s)
            consider(s, fs)
            if fs is None:
                break
            if abs(fs) <= TOL:
                return s

            # Standard secant update: replace the point with larger |f|
            if abs(fs) < abs(fa):
                xa, fa = xb, fb
                xb, fb = s, fs
            else:
                xa, fa = s, fs

            if abs(fs) < abs(ref_f if ref_f is not None else math.inf):
                ref_x, ref_f = s, fs

        # Return best
        if best_x is not None:
            return best_x
        return x0

    # Fallback if we get here (shouldn't happen)
    if best_x is not None:
        return best_x
    return x0
# EVOLVE-BLOCK-END
