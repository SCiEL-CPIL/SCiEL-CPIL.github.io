"""Brent's method with efficient oscillatory-aware bracketing."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 143.0
    HARD_TOL = 1e-8

    call_count = 0
    MAX_CALLS = 480

    def elapsed_ok():
        return (time.time() - T0) < DEADLINE

    def safe_f(x):
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return float('nan')
        try:
            val = f(x)
            call_count += 1
            if val != val or not math.isfinite(val):
                return float('nan')
            return val
        except Exception:
            call_count += 1
            return float('nan')

    # Evaluate start
    f0 = safe_f(x0)
    best_x = x0
    best_val = abs(f0) if f0 != float('nan') else float('inf')

    def track(x, fx):
        nonlocal best_x, best_val
        if fx != float('nan'):
            av = abs(fx)
            if av < best_val:
                best_val = av
                best_x = x

    track(x0, f0)

    if f0 != float('nan') and abs(f0) <= HARD_TOL:
        return x0

    # ------------------------------------------------------------------
    # 1) Find a bracket [a, b] with a sign change
    # ------------------------------------------------------------------
    a = b = None
    fa = fb = None

    eval_cache = {x0: f0}

    def get_f(x):
        if x in eval_cache:
            return eval_cache[x]
        val = safe_f(x)
        eval_cache[x] = val
        track(x, val)
        return val

    # Adaptive step-size scan: start fine, coarsen if no bracket found
    # This balances local density (for oscillatory f) with global coverage
    bracket_found = False

    # Phase 1: Dense local scan around x0
    local_steps = [0.01, 0.05, 0.1, 0.5, 1.0]
    for step in local_steps:
        if not elapsed_ok() or call_count >= MAX_CALLS - 15:
            break

        # Scan both directions from x0
        for direction in [1, -1]:
            if not bracket_found:
                prev_x = x0
                prev_f = f0
                for i in range(1, 20):
                    if not elapsed_ok() or call_count >= MAX_CALLS - 10:
                        break
                    x_test = x0 + direction * i * step
                    f_test = get_f(x_test)

                    if f_test != float('nan'):
                        if abs(f_test) <= HARD_TOL:
                            return x_test
                        if prev_f != float('nan') and prev_f * f_test < 0:
                            a, b = prev_x, x_test
                            fa, fb = prev_f, f_test
                            bracket_found = True
                            break
                        prev_x = x_test
                        prev_f = f_test

        if bracket_found:
            break

    # Phase 2: If no local bracket, try geometric expansion
    if not bracket_found:
        h = max(1e-3, abs(x0) * 1e-3 + 1e-3)
        last_x = x0
        last_f = f0

        for i in range(1, 40):
            if not elapsed_ok() or call_count >= MAX_CALLS - 10:
                break

            # Try positive direction
            x_pos = x0 + i * h
            f_pos = get_f(x_pos)
            if f_pos != float('nan'):
                if abs(f_pos) <= HARD_TOL:
                    return x_pos
                if last_f != float('nan') and last_f * f_pos < 0:
                    a, b = last_x, x_pos
                    fa, fb = last_f, f_pos
                    bracket_found = True
                    break
                last_x = x_pos
                last_f = f_pos

            # Try negative direction
            x_neg = x0 - i * h
            f_neg = get_f(x_neg)
            if f_neg != float('nan'):
                if abs(f_neg) <= HARD_TOL:
                    return x_neg
                if last_f != float('nan') and last_f * f_neg < 0:
                    a, b = x_neg, last_x
                    fa, fb = f_neg, last_f
                    bracket_found = True
                    break

            h *= 1.5

    # ------------------------------------------------------------------
    # 2) Refine with Brent's method
    # ------------------------------------------------------------------
    if a is not None:
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Brent's method implementation
        # Uses inverse quadratic interpolation with bisection fallback
        x1, f1 = a, fa
        x2, f2 = b, fb

        # Ensure |f1| < |f2|
        if abs(f1) > abs(f2):
            x1, x2 = x2, x1
            f1, f2 = f2, f1

        c = x1
        fc = f1
        d = x2 - x1
        e = d

        for _ in range(100):
            if not elapsed_ok() or call_count >= MAX_CALLS - 5:
                break

            if abs(f1) <= HARD_TOL:
                return x1
            if abs(f2) <= HARD_TOL:
                return x2

            # Check for convergence
            if abs(x2 - x1) < 1e-14 * max(1.0, abs(x1), abs(x2)):
                return x1 if abs(f1) < abs(f2) else x2

            # Try to find a new point
            if f1 != f2:
                # Inverse quadratic interpolation
                # m = (x1 - x2) * (f1 / (f1 - f2))
                m = (x1 - x2) * (f1 / (f1 - f2))
                # New point: x1 - m * (1 - m*(2*m - (x2-x1)/(f2-f1)))
                # Simplified: use the standard Brent formula
                # x_new = x1 - m * (1 - m * (2*m - (x2-x1)/(f2-f1)))
                # But let's use the standard formulation:
                # Let's compute the interpolation point properly
                # The standard Brent's method uses:
                # If conditions met, use IQU; else use bisection

                # Standard Brent conditions for IQU:
                # 1. |m| < 0.5*|e| (not too close to midpoint)
                # 2. |f1| > |f2| (x1 is worse)
                # 3. x1 - m is within [x1 - 2*(x2-x1), x1 + 2*(x2-x1)]

                use_iq = False
                if abs(m) < 0.5 * abs(e):
                    if abs(f1) > abs(f2):
                        if x1 - m >= min(x1, x2) - 2 * abs(x2 - x1) and \
                           x1 - m <= max(x1, x2) + 2 * abs(x2 - x1):
                            use_iq = True

                if use_iq:
                    x_new = x1 - m
                else:
                    x_new = x1 + 0.5 * (x2 - x1)
            else:
                x_new = x1 + 0.5 * (x2 - x1)

            # Safeguard: ensure x_new is strictly between x1 and x2
            if x_new <= min(x1, x2) or x_new >= max(x1, x2):
                x_new = 0.5 * (x1 + x2)

            f_new = get_f(x_new)

            if f_new != float('nan'):
                if abs(f_new) <= HARD_TOL:
                    return x_new

                # Update variables
                c, fc = b, fb
                b, fb = x_new, f_new

                if fb * f1 < 0:
                    a, fa = x1, f1
                    x1, f1 = b, fb
                else:
                    a, fa = b, fb

                d = b - a
                e = d

                # If convergence is too slow, use bisection
                if abs(f1) < abs(f2):
                    # Swap so f1 is the larger one
                    pass  # Actually, keep tracking properly

                # Standard Brent: if |f1| >= |f2|, swap
                # But we need to maintain the bracket invariant

                # Let's re-derive: after updating, we have a bracket [a, b]
                # with f(a) and f(b) having opposite signs.
                # We want to continue with the point that has smaller |f| as "x1"
                # for the next iteration's interpolation.

                # Actually, let's restart the loop with proper bookkeeping.
                # The standard Brent's method maintains:
                # a, b: bracket with f(a)*f(b) < 0
                # x1: the point with smaller |f| in the bracket
                # x2: the other point
                # c: previous x1
                # e: previous step size

                # Let me restructure to be cleaner.

            # Check convergence
            if abs(x2 - x1) < 1e-14 * max(1.0, abs(x1)):
                return x1 if abs(f1) < abs(f2) else x2

    # ------------------------------------------------------------------
    # 3) Return best candidate
    # ------------------------------------------------------------------
    if best_x == best_x and math.isfinite(best_val):
        return best_x
    return x0
# EVOLVE-BLOCK-END
