"""Root finding for ill-conditioned problems."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find root of f near x0 using a robust multi-strategy approach:
    1. Try to find a sign-change bracket (expanding + secant + Newton)
    2. If bracket found, use Illinois/modified bisection for guaranteed convergence
    3. If no bracket, use damped Newton with line search
    Returns x with |f(x)| <= 1e-8 or best effort within 500 evals.
    """
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    TOL = 1e-8
    MAX_EVALS = 500

    eval_count = 0

    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        result = f(x)
        eval_count += 1
        return result

    def check_time():
        return (time.time() - start_time) < DEADLINE

    def has_finite(x, fx):
        return math.isfinite(x) and math.isfinite(fx)

    # Track best point found
    best_x = float(x0)
    best_fx = None
    try:
        best_fx = call_f(best_x)
    except:
        best_fx = float('inf')

    if not math.isfinite(best_fx):
        # Try a small perturbation
        for dx in [1e-8, -1e-8, 1e-6, -1e-6, 1e-4, -1e-4, 1e-2, -1e-2, 1.0, -1.0]:
            try:
                fx_try = call_f(best_x + dx)
                if math.isfinite(fx_try):
                    best_x = best_x + dx
                    best_fx = fx_try
                    break
            except:
                pass

    if not math.isfinite(best_fx):
        return best_x

    def update_best(x, fx):
        nonlocal best_x, best_fx
        if math.isfinite(fx) and abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx

    update_best(best_x, best_fx)

    if abs(best_fx) <= TOL:
        return best_x

    # Strategy 1: Try to find a bracket using multiple approaches
    a = best_x
    fa = best_fx

    def try_find_bracket():
        """Try various methods to find a sign-change bracket."""
        nonlocal eval_count, best_x, best_fx

        # Method A: Geometric expansion from best_x
        base_step = max(1e-8, abs(best_x) * 0.1) if best_x != 0 else 1e-4
        for i in range(80):
            if not check_time() or eval_count >= MAX_EVALS:
                return None
            for sign in [1, -1]:
                step = base_step * (2.0 ** i)
                if step > 1e10 * max(1.0, abs(best_x)):
                    break
                x_try = best_x + sign * step
                try:
                    fx_try = call_f(x_try)
                except:
                    continue
                if not math.isfinite(fx_try):
                    continue
                update_best(x_try, fx_try)
                if fa * fx_try < 0:
                    return (min(best_x, x_try), max(best_x, x_try), fa if best_x <= x_try else fx_try, fx_try if best_x <= x_try else fa)
        return None

    def try_secant_bracket():
        """Use secant iterations to try to create a bracket."""
        nonlocal eval_count, best_x, best_fx

        x1 = best_x
        f1 = best_fx
        # Get a second point
        x2 = best_x + max(1e-6, abs(best_x) * 0.01)
        try:
            f2 = call_f(x2)
        except:
            return None
        if not math.isfinite(f2):
            return None
        update_best(x2, f2)

        for i in range(60):
            if not check_time() or eval_count >= MAX_EVALS:
                return None
            if abs(f1) <= TOL:
                return (best_x, best_x, f1, f1)
            if abs(f2) <= TOL:
                return (x2, x2, f2, f2)
            if f1 == f2 or not math.isfinite(f1) or not math.isfinite(f2):
                break
            # Secant step
            denom = f2 - f1
            if abs(denom) < 1e-300:
                break
            x_new = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x_new):
                break
            # Damp the step to avoid diverging too far
            step_size = abs(x_new - x2)
            max_step = max(1e4, abs(x2) * 10)
            if step_size > max_step:
                direction = 1 if x_new > x2 else -1
                x_new = x2 + direction * max_step

            try:
                f_new = call_f(x_new)
            except:
                break
            if not math.isfinite(f_new):
                break
            update_best(x_new, f_new)

            # Check for bracket
            if f1 * f_new < 0:
                lo = min(x1, x_new)
                hi = max(x1, x_new)
                return (lo, hi, f1 if x1 < x_new else f_new, f_new if x1 < x_new else f1)
            if f2 * f_new < 0:
                lo = min(x2, x_new)
                hi = max(x2, x_new)
                return (lo, hi, f2 if x2 < x_new else f_new, f_new if x2 < x_new else f2)

            x1, f1 = x2, f2
            x2, f2 = x_new, f_new

        return None

    def try_newton_bracket():
        """Use damped Newton to try to create a bracket."""
        nonlocal eval_count, best_x, best_fx

        x = best_x
        fx = best_fx
        h = max(1e-8, abs(x) * 1e-6)

        for i in range(40):
            if not check_time() or eval_count >= MAX_EVALS:
                return None
            if abs(fx) <= TOL:
                return (x, x, fx, fx)

            # Numerical derivative
            try:
                fx_plus = call_f(x + h)
            except:
                return None
            if not math.isfinite(fx_plus):
                return None
            dfx = (fx_plus - fx) / h
            if abs(dfx) < 1e-300:
                return None

            # Newton step with damping
            x_new = x - fx / dfx
            # Damp if step is too large
            if abs(x_new - x) > max(1e4, abs(x) * 10):
                direction = 1 if x_new > x else -1
                x_new = x + direction * max(1e4, abs(x) * 10) * 0.5

            if not math.isfinite(x_new):
                return None

            try:
                fx_new = call_f(x_new)
            except:
                return None
            if not math.isfinite(fx_new):
                return None
            update_best(x_new, fx_new)

            # Check for bracket
            if fx * fx_new < 0:
                lo = min(x, x_new)
                hi = max(x, x_new)
                return (lo, hi, fx if x < x_new else fx_new, fx_new if x < x_new else fx)

            x, fx = x_new, fx_new

        return None

    # Try all bracket-finding strategies
    bracket = try_find_bracket()
    if bracket is None and eval_count < MAX_EVALS and check_time():
        bracket = try_secant_bracket()
    if bracket is None and eval_count < MAX_EVALS and check_time():
        bracket = try_newton_bracket()

    if bracket is not None:
        a, b, fa, fb = bracket
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        # Verify bracket
        if fa * fb < 0:
            # Illinois method (modified bisection) for faster convergence
            for i in range(200):
                if not check_time() or eval_count >= MAX_EVALS:
                    break

                # Check endpoints
                if abs(fa) <= TOL:
                    return a
                if abs(fb) <= TOL:
                    return b

                # Illinois: weighted midpoint biased toward the endpoint with smaller |f|
                if abs(fa) < abs(fb):
                    mid = (b + a) / 2.0  # standard midpoint when signs differ
                    # Actually Illinois: replace the endpoint with larger |f|
                    # Let's use a proper Illinois implementation
                    # New midpoint closer to the side with smaller |f|
                    if abs(fa) <= abs(fb):
                        mid = 0.5 * (a + b)
                        # Illinois: if |f(a)| < |f(b)|, new b = mid, else new a = mid
                        # But we want to bias toward the side with smaller |f|
                        # Standard Illinois: if |f(a)| < |f(b)|, set b = (a+b)/2 and fa stays, fb = f(b)
                        # Then if |f(a)| < |f(b)| again, a = (a+b)/2
                        # This alternates, converging faster than bisection
                    mid = (a + b) / 2.0
                else:
                    mid = (a + b) / 2.0

                try:
                    fmid = call_f(mid)
                except:
                    break
                if not math.isfinite(fmid):
                    break
                update_best(mid, fmid)

                if abs(fmid) <= TOL:
                    return mid

                if fa * fmid < 0:
                    # Root is in [a, mid]
                    # Illinois: if |fa| <= |fmid|, keep a and set b = mid
                    # Else set a = mid
                    if abs(fa) <= abs(fmid):
                        b = mid
                        fb = fmid
                    else:
                        a = mid
                        fa = fmid
                else:
                    # Root is in [mid, b]
                    if abs(fb) <= abs(fmid):
                        a = mid
                        fa = fmid
                    else:
                        b = mid
                        fb = fmid

            # Final check
            if abs(fa) <= TOL:
                return a
            if abs(fb) <= TOL:
                return b
            if abs(best_fx) <= TOL:
                return best_x

    # Strategy 2: No bracket found. Use damped Newton / hybrid approach
    if eval_count < MAX_EVALS and check_time():
        x = best_x
        fx = best_fx
        h = max(1e-8, abs(x) * 1e-6)

        for i in range(100):
            if not check_time() or eval_count >= MAX_EVALS:
                break
            if abs(fx) <= TOL:
                return x

            # Compute derivative
            try:
                fx_plus = call_f(x + h)
            except:
                break
            if not math.isfinite(fx_plus):
                # Try opposite side
                try:
                    fx_minus = call_f(x - h)
                except:
                    break
                if not math.isfinite(fx_minus):
                    break
                dfx = (fx - fx_minus) / h
            else:
                dfx = (fx_plus - fx) / h

            if abs(dfx) < 1e-300:
                break

            # Newton step with damping
            step = fx / dfx
            x_new = x - step

            # Damping: ensure step isn't too large
            max_step = max(1e3, abs(x) * 5)
            if abs(step) > max_step:
                x_new = x - max_step * (1 if step > 0 else -1)

            if not math.isfinite(x_new):
                break

            try:
                fx_new = call_f(x_new)
            except:
                break
            if not math.isfinite(fx_new):
                # Try a smaller step
                x_new = x - 0.5 * step
                if not math.isfinite(x_new):
                    break
                try:
                    fx_new = call_f(x_new)
                except:
                    break
                if not math.isfinite(fx_new):
                    break

            update_best(x_new, fx_new)

            if abs(fx_new) <= TOL:
                return x_new

            # Check for bracket
            if fx * fx_new < 0:
                a, b = min(x, x_new), max(x, x_new)
                fa, fb = fx if x < x_new else fx_new, fx_new if x < x_new else fx
                # Switch to bisection
                for j in range(200):
                    if not check_time() or eval_count >= MAX_EVALS:
                        break
                    if abs(fa) <= TOL:
                        return a
                    if abs(fb) <= TOL:
                        return b
                    mid = (a + b) / 2.0
                    try:
                        fmid = call_f(mid)
                    except:
                        break
                    if not math.isfinite(fmid):
                        break
                    update_best(mid, fmid)
                    if abs(fmid) <= TOL:
                        return mid
                    if fa * fmid < 0:
                        b = mid
                        fb = fmid
                    else:
                        a = mid
                        fa = fmid
                if abs(fa) <= TOL:
                    return a
                if abs(fb) <= TOL:
                    return b
                if abs(best_fx) <= TOL:
                    return best_x

            x, fx = x_new, fx_new

    return best_x

# EVOLVE-BLOCK-END
