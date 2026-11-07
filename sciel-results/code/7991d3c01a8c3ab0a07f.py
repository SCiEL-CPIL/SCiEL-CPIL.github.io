"""Root finder for 3 fixed oscillatory equations using adaptive multi-scale grid + Illinois method."""
import math
import time


# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 117.0  # Safety margin for 120s limit (evaluator kills at 150s, but we want margin)
    MAX_CALLS = 495     # Stay under 500

    calls = 0
    best_x = float(x0)
    best_abs_f = float('inf')

    def _time_exceeded():
        return time.time() - start_time > TIME_LIMIT

    def _f(x):
        nonlocal calls, best_x, best_abs_f
        if calls >= MAX_CALLS:
            return float('nan')
        if _time_exceeded():
            return float('nan')
        calls += 1
        try:
            val = f(x)
        except Exception:
            return float('nan')
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            return float('nan')
        abs_val = abs(val)
        if abs_val < best_abs_f:
            best_abs_f = abs_val
            best_x = float(x)
        return float(val)

    # Evaluate at x0
    x = float(x0)
    fx = _f(x)
    if math.isnan(fx):
        return best_x
    if abs(fx) <= 1e-8:
        return x

    # Illinois method for root refinement
    def _illinois(a, b, fa, fb, max_iter=60):
        """Illinois method to find root in [a, b] where f(a)*f(b) < 0."""
        nonlocal calls, best_x, best_abs_f
        if fa * fb > 0:
            return None
        for _ in range(max_iter):
            if calls >= MAX_CALLS - 5 or _time_exceeded():
                break
            # Illinois update: replace the endpoint with the larger |f|
            if abs(fa) > abs(fb):
                # Interpolate towards a
                c = (a * fb - b * fa) / (fb - fa)
                fc = _f(c)
                if math.isnan(fc):
                    break
                if abs(fc) <= 1e-8:
                    return c
                if fa * fc < 0:
                    b, fb = c, fc
                else:
                    a, fa = c, fc
                    # Illinois: shrink the side we're moving away from
                    a = (a + b) / 2
                    fa = _f(a)
                    if math.isnan(fa):
                        break
            else:
                # Interpolate towards b
                c = (a * fb - b * fa) / (fb - fa)
                fc = _f(c)
                if math.isnan(fc):
                    break
                if abs(fc) <= 1e-8:
                    return c
                if fb * fc < 0:
                    a, fa = c, fc
                else:
                    b, fb = c, fc
                    # Illinois: shrink the side we're moving away from
                    b = (a + b) / 2
                    fb = _f(b)
                    if math.isnan(fb):
                        break
            if abs(b - a) < 1e-14:
                break
        # Return the point with smaller |f|
        if abs(fa) < abs(fb):
            return a
        else:
            return b

    # Multi-scale grid scan to find a bracket
    # Strategy: scan at multiple scales, starting from small radius near x0
    # and expanding outward. Use adaptive density.

    bracket_found = False
    bracket = None

    # Define scan ranges: (radius, num_intervals)
    # We want to catch oscillatory sign changes.
    # Start with fine grid near x0, then coarser grids over larger ranges.
    scan_configs = [
        (0.5, 50),    # Step ~0.02
        (1.0, 80),    # Step ~0.025
        (2.0, 100),   # Step ~0.04
        (5.0, 100),   # Step ~0.1
        (10.0, 100),  # Step ~0.2
        (20.0, 80),   # Step ~0.5
        (50.0, 60),   # Step ~1.67
        (100.0, 50),  # Step ~4.0
    ]

    for radius, n_intervals in scan_configs:
        if calls >= MAX_CALLS - 30 or _time_exceeded():
            break

        lower = x - radius
        upper = x + radius

        prev_x = None
        prev_f = None

        for i in range(n_intervals + 1):
            if calls >= MAX_CALLS - 30 or _time_exceeded():
                break
            cur_x = lower + i * (upper - lower) / n_intervals

            # Skip if too close to a previously evaluated point
            if prev_x is not None and abs(cur_x - prev_x) < 1e-15:
                continue

            cur_f = _f(cur_x)
            if math.isnan(cur_f):
                prev_x = cur_x
                prev_f = cur_f
                continue

            if abs(cur_f) <= 1e-8:
                return cur_x

            if prev_x is not None and prev_f is not None and not math.isnan(prev_f):
                if prev_f * cur_f < 0:
                    bracket = (prev_x, cur_x)
                    bracket_found = True
                    break

            prev_x = cur_x
            prev_f = cur_f

        if bracket_found:
            break

    # If a bracket was found, use Illinois method to refine
    if bracket_found:
        a, b = bracket
        fa = _f(a)
        fb = _f(b)
        if not math.isnan(fa) and not math.isnan(fb) and fa * fb < 0:
            root = _illinois(a, b, fa, fb)
            if root is not None and math.isfinite(root):
                f_root = _f(root)
                if not math.isnan(f_root) and abs(f_root) <= 1e-8:
                    return root
                if not math.isnan(f_root) and abs(f_root) < best_abs_f:
                    best_x = root
                    best_abs_f = abs(f_root)

    # If no root found via bracketing, try a finer local search around the best point
    if best_abs_f > 1e-8 and calls < MAX_CALLS - 20 and not _time_exceeded():
        # Local scan around best_x with very fine grid
        local_radius = 0.1
        local_n = 40
        lower = best_x - local_radius
        upper = best_x + local_radius

        prev_x = None
        prev_f = None

        for i in range(local_n + 1):
            if calls >= MAX_CALLS - 10 or _time_exceeded():
                break
            cur_x = lower + i * (upper - lower) / local_n
            if prev_x is not None and abs(cur_x - prev_x) < 1e-15:
                continue

            cur_f = _f(cur_x)
            if math.isnan(cur_f):
                prev_x = cur_x
                prev_f = cur_f
                continue

            if abs(cur_f) <= 1e-8:
                return cur_x

            if prev_x is not None and prev_f is not None and not math.isnan(prev_f):
                if prev_f * cur_f < 0:
                    root = _illinois(prev_x, cur_x, prev_f, cur_f)
                    if root is not None and math.isfinite(root):
                        f_root = _f(root)
                        if not math.isnan(f_root) and abs(f_root) <= 1e-8:
                            return root
                        if not math.isnan(f_root) and abs(f_root) < best_abs_f:
                            best_x = root
                            best_abs_f = abs(f_root)
                    break

            prev_x = cur_x
            prev_f = cur_f

    return best_x

# EVOLVE-BLOCK-END
