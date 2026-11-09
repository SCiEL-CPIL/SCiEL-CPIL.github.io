"""Derivative-free Illinois/bisection with dense oscillatory-aware bracketing."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 143.0          # safety margin before 150 s hard kill
    TOL = 1e-12               # aim well below 1e-8
    HARD_TOL = 1e-8           # contract tolerance
    MAX_REFINE_ITERS = 300

    call_count = 0
    MAX_CALLS = 450           # Hard limit on calls to ensure safety

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

    # Strategy: Scan outward from x0 with increasing step sizes.
    # For oscillatory functions, a dense local scan is better than pure geometric expansion
    # because roots might be frequent.
    # We will try a sequence of steps: 1e-3, 1e-2, 1e-1, 1, 10, 100 ...
    # But also check intermediate points to avoid skipping.

    # Let's use a hybrid:
    # 1. Local fine scan: [x0 - 0.1, x0 + 0.1] with step 0.01 (21 points)
    # 2. If no bracket, expand to [x0 - 1, x0 + 1] with step 0.1 (21 points)
    # 3. If no bracket, expand to [x0 - 10, x0 + 10] with step 1 (21 points)
    # 4. If no bracket, geometric expansion.

    scales = [0.1, 1.0, 10.0, 100.0, 1000.0]
    bracket_found = False

    for scale in scales:
        if not elapsed_ok() or call_count >= MAX_CALLS - 20:
            break

        # Scan from x0 - scale to x0 + scale
        # We need to handle the case where x0 is not in the middle of the grid.
        # Simpler: Just check points x0 + k * (scale/10) for k in -10..10

        step = scale / 10.0

        prev_x = None
        prev_f = None

        for k in range(-10, 11):
            if not elapsed_ok() or call_count >= MAX_CALLS - 10:
                break

            x_test = x0 + k * step
            f_test = get_f(x_test)

            if f_test != float('nan'):
                if abs(f_test) <= HARD_TOL:
                    return x_test

                if prev_x is not None and prev_f is not None:
                    if prev_f * f_test < 0:
                        a, b = prev_x, x_test
                        fa, fb = prev_f, f_test
                        bracket_found = True
                        break
                prev_x = x_test
                prev_f = f_test

            if bracket_found:
                break

        if bracket_found:
            break

    # If still no bracket, try geometric expansion in both directions
    if not bracket_found:
        h = max(1e-4, abs(x0) * 1e-4)
        last_pos_x = x0
        last_pos_f = f0
        last_neg_x = x0
        last_neg_f = f0

        for i in range(1, 30):
            if not elapsed_ok() or call_count >= MAX_CALLS - 10:
                break

            # Positive side
            x_new_pos = x0 + i * h
            f_new_pos = get_f(x_new_pos)

            if f_new_pos != float('nan') and last_pos_f != float('nan'):
                if f_new_pos * last_pos_f < 0:
                    a, b = last_pos_x, x_new_pos
                    fa, fb = last_pos_f, f_new_pos
                    bracket_found = True
                    break
                elif abs(f_new_pos) <= HARD_TOL:
                    return x_new_pos

            # Update frontier
            if f_new_pos != float('nan'):
                last_pos_x = x_new_pos
                last_pos_f = f_new_pos

            # Negative side
            x_new_neg = x0 - i * h
            f_new_neg = get_f(x_new_neg)

            if f_new_neg != float('nan') and last_neg_f != float('nan'):
                if f_new_neg * last_neg_f < 0:
                    a, b = x_new_neg, last_pos_x
                    fa, fb = f_new_neg, last_pos_f
                    bracket_found = True
                    break
                elif abs(f_new_neg) <= HARD_TOL:
                    return x_new_neg

            # Update frontier
            if f_new_neg != float('nan'):
                last_neg_x = x_new_neg
                last_neg_f = f_new_neg

            # Increase step size
            h *= 2.0

    # ------------------------------------------------------------------
    # 2) Refine with Illinois Algorithm
    # ------------------------------------------------------------------
    if a is not None:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Illinois algorithm is a variant of bisection that uses a secant-like step
        # but shrinks the larger interval.

        # Initialize
        x1, f1 = a, fa
        x2, f2 = b, fb

        # Ensure |f1| < |f2| so x1 is the "better" point
        if abs(f1) > abs(f2):
            x1, x2 = x2, x1
            f1, f2 = f2, f1

        for _ in range(MAX_REFINE_ITERS):
            if not elapsed_ok() or call_count >= MAX_CALLS - 5:
                break

            if abs(f1) <= TOL:
                return x1
            if abs(f2) <= TOL:
                return x2

            # Calculate Illinois point
            # x_new = (x1*f2 - x2*f1) / (f2 - f1)
            # This is the secant point.
            # But Illinois modifies this:
            # If x_new is outside [x1, x2], it is not used directly.
            # Instead, the interval with the larger absolute function value is halved.

            # Standard Illinois:
            # 1. Compute x_new via secant.
            # 2. If x_new is not in (x1, x2), set x_new = (x1+x2)/2 (bisection).
            # 3. Evaluate f(x_new).
            # 4. If f(x_new) has same sign as f1, replace x1 with x_new.
            #    Else replace x2 with x_new.
            # 5. If the replaced point was the one with larger |f|, halve its |f| value.

            if f2 - f1 == 0:
                x_new = 0.5 * (x1 + x2)
            else:
                x_new = (x1 * f2 - x2 * f1) / (f2 - f1)

            # Safeguard: if x_new is not finite or outside bracket, use bisection
            if not math.isfinite(x_new) or x_new <= x1 or x_new >= x2:
                x_new = 0.5 * (x1 + x2)

            f_new = get_f(x_new)

            if f_new != float('nan'):
                if abs(f_new) <= TOL:
                    return x_new

                # Determine which point to replace
                if f_new * f1 < 0:
                    # Root is between x_new and x1
                    # Replace x2
                    if abs(f2) > abs(f1):
                        # f2 was the larger one, so halve it
                        f2 *= 0.5
                    x2 = x_new
                    f2 = f_new
                else:
                    # Root is between x_new and x2
                    # Replace x1
                    if abs(f1) > abs(f2):
                        # f1 was the larger one, so halve it
                        f1 *= 0.5
                    x1 = x_new
                    f1 = f_new

                # Re-order if necessary? Illinois doesn't strictly require ordering by |f|
                # but it helps convergence.
                # Actually, standard Illinois keeps x1 as the "best" point.
                # If we replaced x1, we might need to swap.
                # Let's just ensure we track the best point for return.
                track(x_new, f_new)

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
