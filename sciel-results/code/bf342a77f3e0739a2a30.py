"""Root finder for 3 fixed oscillatory equations using bracketing + Illinois method."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin for 120s limit
    MAX_CALLS = 480     # Stay well under 500

    calls = 0
    best_x = float(x0)
    best_abs_f = float('inf')

    def _f(x):
        nonlocal calls, best_x, best_abs_f
        if calls >= MAX_CALLS:
            return float('nan')
        if time.time() - start_time > TIME_LIMIT:
            return float('nan')
        calls += 1
        val = f(x)
        abs_val = abs(val)
        if abs_val < best_abs_f:
            best_abs_f = abs_val
            best_x = x
        return val

    def illinois(fa, a, fb, b):
        """Illinois method: bisection + linear interpolation."""
        nonlocal calls
        for _ in range(200):
            if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
                break
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            # Illinois: linear interpolation, but if it's outside [a,b], use midpoint
            if fa == fb:
                mid = (a + b) / 2.0
            else:
                mid = (a * fb - b * fa) / (fb - fa)
            if not (a < mid < b) or not math.isfinite(mid):
                mid = (a + b) / 2.0
            fm = _f(mid)
            if math.isnan(fm):
                break
            if abs(fm) <= 1e-8:
                return mid
            if fa * fm < 0:
                # Root in [a, mid]
                b = mid
                fb = fm
                # Illinois: shrink fa to encourage bisection
                fa = fa / 2.0
            else:
                # Root in [mid, b]
                a = mid
                fa = fm
                # Illinois: shrink fb to encourage bisection
                fb = fb / 2.0
        # Return the better endpoint
        if abs(fa) <= abs(fb):
            return a
        return b

    # Phase 1: Find a bracket around x0
    x = float(x0)
    fx = _f(x)
    if math.isnan(fx):
        return best_x
    if abs(fx) <= 1e-8:
        return x

    # Try expanding in both directions to find a sign change
    step = 0.1 * max(1.0, abs(x))
    found_bracket = False

    # Try positive direction
    for i in range(1, 50):
        if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
            break
        x_pos = x + i * step
        f_pos = _f(x_pos)
        if math.isnan(f_pos):
            break
        if abs(f_pos) <= 1e-8:
            return x_pos
        if fx * f_pos < 0:
            # Found bracket [x, x_pos]
            result = illinois(fx, x, f_pos, x_pos)
            return result
        step *= 1.5  # Expand adaptively

    # Try negative direction
    step = 0.1 * max(1.0, abs(x))
    for i in range(1, 50):
        if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
            break
        x_neg = x - i * step
        f_neg = _f(x_neg)
        if math.isnan(f_neg):
            break
        if abs(f_neg) <= 1e-8:
            return x_neg
        if fx * f_neg < 0:
            # Found bracket [x_neg, x]
            result = illinois(f_neg, x_neg, fx, x)
            return result
        step *= 1.5

    # Phase 2: If no bracket found, do a finer local scan
    # Try a dense local grid around x0
    local_step = 0.01 * max(1.0, abs(x))
    prev_x = x - 10 * local_step
    prev_fx = _f(prev_x)
    if not math.isnan(prev_fx) and abs(prev_fx) <= 1e-8:
        return prev_x

    for i in range(21):
        if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
            break
        cur_x = x - 10 * local_step + i * local_step
        cur_fx = _f(cur_x)
        if math.isnan(cur_fx):
            break
        if abs(cur_fx) <= 1e-8:
            return cur_x
        if not math.isnan(prev_fx) and prev_fx * cur_fx < 0:
            result = illinois(prev_fx, prev_x, cur_fx, cur_x)
            return result
        prev_x = cur_x
        prev_fx = cur_fx

    # Phase 3: Newton polish with finite-difference derivative
    # Use the best_x found so far and try to polish
    x_best = best_x
    f_best = best_abs_f
    if math.isfinite(f_best) and f_best < 1.0:
        for _ in range(50):
            if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
                break
            h = max(1e-8, 1e-6 * max(1.0, abs(x_best)))
            fp = _f(x_best + h)
            if math.isnan(fp):
                break
            fm = _f(x_best - h)
            if math.isnan(fm):
                break
            if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
                break
            df = (fp - fm) / (2.0 * h)
            if abs(df) < 1e-30:
                break
            x_new = x_best - f_best / df
            if not math.isfinite(x_new):
                break
            # Damped Newton: if step is too large, reduce it
            if abs(x_new - x_best) > 10.0 * max(1.0, abs(x_best)):
                x_new = x_best + 0.5 * max(1.0, abs(x_best)) * (1.0 if df > 0 else -1.0)
            f_new = _f(x_new)
            if math.isnan(f_new):
                break
            if abs(f_new) <= 1e-8:
                return x_new
            if abs(f_new) < f_best:
                x_best = x_new
                f_best = abs(f_new)
            else:
                # Halve the step
                x_new = x_best - 0.5 * f_best / df
                f_new = _f(x_new)
                if math.isnan(f_new):
                    break
                if abs(f_new) <= 1e-8:
                    return x_new
                if abs(f_new) < f_best:
                    x_best = x_new
                    f_best = abs(f_new)

    # Return best x found
    return best_x

# EVOLVE-BLOCK-END
