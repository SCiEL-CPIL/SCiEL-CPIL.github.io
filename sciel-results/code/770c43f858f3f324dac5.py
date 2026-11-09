"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f near x0 using a robust hybrid method:
    - Start with Newton-Raphson (fast for smooth polynomials)
    - If Newton stalls or diverges, fall back to bisection on a bracketed interval
    - Use finite-difference derivative
    - Track best x seen (smallest |f(x)|) and return it at the end
    - Deadline guard: return best if we've been running too long
    - Call counter to stay under 500 calls
    """
    DEADLINE = 150.0  # seconds
    MAX_CALLS = 450    # safety margin under 500
    start_time = time.time()
    call_count = 0

    def fcounted(x_val):
        nonlocal call_count
        call_count += 1
        return f(x_val)

    x = float(x0)
    best_x = x
    best_fx = fcounted(x)
    best_abs_fx = abs(best_fx)

    # If we already have a root, return immediately
    if best_abs_fx <= 1e-8:
        return best_x

    # Try to bracket the root using expanding intervals from x0
    def try_bracket(xc, fa_val):
        """Try to find a bracket [a, b] where f(a)*f(b) <= 0 near xc."""
        if abs(fa_val) <= 1e-8:
            return xc, xc
        # Expand in both directions
        step = max(1e-4, abs(xc) * 1e-4)
        # Try expanding to the right
        for i in range(1, 40):
            if time.time() - start_time > DEADLINE - 5 or call_count >= MAX_CALLS - 10:
                break
            x_try = xc + step * i
            fx_try = fcounted(x_try)
            if abs(fx_try) <= 1e-8:
                return x_try, x_try
            if fa_val * fx_try < 0:
                return xc, x_try
        # Try expanding to the left
        for i in range(1, 40):
            if time.time() - start_time > DEADLINE - 5 or call_count >= MAX_CALLS - 10:
                break
            x_try = xc - step * i
            fx_try = fcounted(x_try)
            if abs(fx_try) <= 1e-8:
                return x_try, x_try
            if fa_val * fx_try < 0:
                return x_try, xc
        return None, None

    # Phase 1: Newton's method with safeguards
    x_newton = x
    fx_newton = best_fx
    for it in range(60):
        if time.time() - start_time > DEADLINE - 5 or call_count >= MAX_CALLS - 10:
            break
        if abs(fx_newton) <= 1e-12:
            return x_newton
        h = 1e-7 * max(1.0, abs(x_newton))
        fx_plus = fcounted(x_newton + h)
        d = (fx_plus - fx_newton) / h
        if d == 0 or not math.isfinite(d):
            break
        x_candidate = x_newton - fx_newton / d
        if not math.isfinite(x_candidate):
            break
        # Damping: don't move too far
        max_step = max(1.0, abs(x_newton) * 0.5)
        if abs(x_candidate - x_newton) > max_step:
            direction = 1.0 if x_candidate > x_newton else -1.0
            x_candidate = x_newton + direction * max_step
        fx_candidate = fcounted(x_candidate)
        if not math.isfinite(fx_candidate):
            break
        # Update best
        if abs(fx_candidate) < best_abs_fx:
            best_x = x_candidate
            best_fx = fx_candidate
            best_abs_fx = abs(fx_candidate)
        if best_abs_fx <= 1e-8:
            return best_x
        # Accept if it's making progress
        if abs(fx_candidate) < abs(fx_newton) * 1.5:
            x_newton = x_candidate
            fx_newton = fx_candidate
        else:
            # Halve the step
            x_candidate = (x_newton + x_newton - fx_newton / d) / 2.0
            fx_candidate = fcounted(x_candidate)
            if not math.isfinite(fx_candidate):
                break
            if abs(fx_candidate) < best_abs_fx:
                best_x = x_candidate
                best_fx = fx_candidate
                best_abs_fx = abs(fx_candidate)
            if best_abs_fx <= 1e-8:
                return best_x
            x_newton = x_candidate
            fx_newton = fx_candidate

    # Phase 2: Try to bracket and use bisection
    if time.time() - start_time < DEADLINE - 5 and call_count < MAX_CALLS - 10:
        a, b = try_bracket(x, best_fx)
        if a is not None and a != b:
            fa = fcounted(a)
            fb = fcounted(b)
            if abs(fa) < best_abs_fx:
                best_x = a
                best_abs_fx = abs(fa)
            if abs(fb) < best_abs_fx:
                best_x = b
                best_abs_fx = abs(fb)
            if best_abs_fx <= 1e-8:
                return best_x
            # Bisection
            for it in range(60):
                if time.time() - start_time > DEADLINE - 2 or call_count >= MAX_CALLS - 5:
                    break
                mid = (a + b) / 2.0
                fmid = fcounted(mid)
                if not math.isfinite(fmid):
                    break
                if abs(fmid) < best_abs_fx:
                    best_x = mid
                    best_abs_fx = abs(fmid)
                if best_abs_fx <= 1e-8:
                    return best_x
                if fa * fmid <= 0:
                    b = mid
                    fb = fmid
                else:
                    a = mid
                    fa = fmid
                if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                    break

    # Phase 3: Final high-precision Newton polish from best_x
    if time.time() - start_time < DEADLINE - 1 and call_count < MAX_CALLS - 5:
        x_ref = best_x
        for it in range(30):
            if time.time() - start_time > DEADLINE - 0.5 or call_count >= MAX_CALLS:
                break
            fx_ref = fcounted(x_ref)
            if not math.isfinite(fx_ref):
                break
            if abs(fx_ref) < best_abs_fx:
                best_x = x_ref
                best_abs_fx = abs(fx_ref)
            if best_abs_fx <= 1e-10:
                return best_x
            h = 1e-8 * max(1.0, abs(x_ref))
            fx_plus = fcounted(x_ref + h)
            d = (fx_plus - fx_ref) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x_ref - fx_ref / d
            if not math.isfinite(x_new):
                break
            # Limit step
            max_step = max(1e-4, abs(x_ref) * 0.1)
            if abs(x_new - x_ref) > max_step:
                direction = 1.0 if x_new > x_ref else -1.0
                x_new = x_ref + direction * max_step
            x_ref = x_new

    return best_x
# EVOLVE-BLOCK-END
