"""Root finding for ill-conditioned equations: adaptive multi-method with local linearization."""
import time
import math

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations of f.
    Returns the best x found (minimizing |f(x)|) within the budget.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    max_evals = 500
    evals_used = 0
    best_x = float(x0)
    best_fx = None

    def check_time():
        return time.time() < deadline

    def safe_eval(x):
        nonlocal evals_used
        if not check_time():
            return None
        if evals_used >= max_evals:
            return None
        if not math.isfinite(x):
            return None
        try:
            val = f(x)
            evals_used += 1
            if not isinstance(val, (int, float)):
                return None
            if math.isnan(val) or math.isinf(val):
                return None
            return float(val)
        except Exception:
            evals_used += 1
            return None

    def update_best(x, fx):
        nonlocal best_x, best_fx
        if best_fx is None or abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
        return best_fx

    # Initial evaluation
    fx0 = safe_eval(best_x)
    if fx0 is not None:
        update_best(best_x, fx0)
        if abs(best_fx) <= 1e-8:
            return best_x

    # If initial eval failed, try a few perturbations
    if best_fx is None:
        for dx in [1e-8, -1e-8, 1e-4, -1e-4, 0.01, -0.01, 0.1, -0.1, 1.0, -1.0, 10.0, -10.0]:
            if not check_time() or evals_used >= max_evals:
                break
            trial_x = float(x0) + dx
            trial_fx = safe_eval(trial_x)
            if trial_fx is not None:
                update_best(trial_x, trial_fx)
                if abs(best_fx) <= 1e-8:
                    return best_x
        if best_fx is None:
            return best_x

    # ============================================================
    # Strategy 1: Secant method with adaptive step
    # ============================================================
    def try_secant(x0_val, fx0_val, max_iters=200):
        nonlocal evals_used
        if not check_time() or evals_used >= max_evals - 2:
            return False, None, None

        scale = max(1.0, abs(x0_val))
        x1 = x0_val + 1e-6 * scale
        fx1 = safe_eval(x1)
        if fx1 is None:
            x1 = x0_val - 1e-6 * scale
            fx1 = safe_eval(x1)
        if fx1 is None:
            for dx in [1e-4, -1e-4, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
                x1 = x0_val + dx
                fx1 = safe_eval(x1)
                if fx1 is not None:
                    break
        if fx1 is None:
            return False, None, None

        update_best(x1, fx1)
        if abs(fx1) <= 1e-8:
            return True, x1, fx1

        x_prev, fx_prev = x0_val, fx0_val
        x_curr, fx_curr = x1, fx1

        for _ in range(max_iters):
            if not check_time() or evals_used >= max_evals - 2:
                return False, x_curr, fx_curr
            if abs(fx_curr) <= 1e-8:
                return True, x_curr, fx_curr
            if abs(fx_prev) <= 1e-8:
                return True, x_prev, fx_prev

            denom = fx_curr - fx_prev
            if abs(denom) < 1e-18:
                break

            x_new = x_curr - fx_curr * (x_curr - x_prev) / denom

            if not math.isfinite(x_new):
                break

            # Adaptive step limiting
            max_step = 10.0 * max(1.0, abs(x_curr))
            if abs(x_new - x_curr) > max_step:
                x_new = x_curr - math.copysign(max_step, x_new - x_curr)

            fx_new = safe_eval(x_new)
            if fx_new is None:
                break

            update_best(x_new, fx_new)
            if abs(fx_new) <= 1e-8:
                return True, x_new, fx_new

            # Check for convergence
            if abs(x_new - x_curr) < 1e-14 * max(1.0, abs(x_new)):
                return False, x_new, fx_new

            x_prev, fx_prev = x_curr, fx_curr
            x_curr, fx_curr = x_new, fx_new

        return False, x_curr, fx_curr

    success, secant_x, secant_fx = try_secant(best_x, best_fx, 200)
    if success:
        return secant_x

    # Update best if secant made progress
    if secant_fx is not None:
        update_best(secant_x, secant_fx)
        if abs(best_fx) <= 1e-8:
            return best_x

    # ============================================================
    # Strategy 2: Bracket search + bisection
    # ============================================================
    if abs(best_fx) > 1e-8 and check_time() and evals_used < max_evals - 10:
        bracket_found = False
        a, b = None, None
        fa, fb = None, None

        base_scale = max(1.0, abs(best_x))
        scales = [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]

        for scale in scales:
            if not check_time() or evals_used >= max_evals - 5:
                break
            delta = scale * base_scale
            xa = best_x - delta
            xb = best_x + delta

            fa = safe_eval(xa)
            if fa is None:
                continue
            fb = safe_eval(xb)
            if fb is None:
                continue

            update_best(xa, fa)
            update_best(xb, fb)
            if abs(best_fx) <= 1e-8:
                return best_x

            if fa * fb < 0:
                a, b = xa, xb
                bracket_found = True
                break

        if bracket_found and check_time() and evals_used < max_evals - 5:
            # Bisection with secant acceleration
            for _ in range(100):
                if not check_time() or evals_used >= max_evals - 2:
                    break
                mid = 0.5 * (a + b)
                fm = safe_eval(mid)
                if fm is None:
                    break

                update_best(mid, fm)
                if abs(fm) <= 1e-8:
                    return mid

                if fa * fm < 0:
                    b = mid
                    fb = fm
                else:
                    a = mid
                    fa = fm

                if abs(b - a) < 1e-14 * max(1.0, abs(mid)):
                    break

    # ============================================================
    # Strategy 3: Local linearization via finite difference + Newton
    # ============================================================
    if abs(best_fx) > 1e-8 and check_time() and evals_used < max_evals - 20:
        x = best_x
        fx = best_fx
        h = 1e-8 * max(1.0, abs(x))

        for _ in range(50):
            if not check_time() or evals_used >= max_evals - 2:
                break
            if abs(fx) <= 1e-8:
                return x

            # Finite difference derivative
            x_plus = x + h
            fx_plus = safe_eval(x_plus)
            if fx_plus is None:
                h *= 0.1
                continue

            x_minus = x - h
            fx_minus = safe_eval(x_minus)
            if fx_minus is None:
                # Forward difference
                df = (fx_plus - fx) / h
            else:
                # Central difference
                df = (fx_plus - fx_minus) / (2.0 * h)

            if abs(df) < 1e-18:
                break

            # Newton step
            x_new = x - fx / df
            if not math.isfinite(x_new):
                break

            # Limit step
            max_step = 5.0 * max(1.0, abs(x))
            if abs(x_new - x) > max_step:
                x_new = x - math.copysign(max_step, x_new - x)

            fx_new = safe_eval(x_new)
            if fx_new is None:
                break

            update_best(x_new, fx_new)
            if abs(fx_new) <= 1e-8:
                return x_new

            # Check convergence
            if abs(x_new - x) < 1e-14 * max(1.0, abs(x_new)):
                break

            x = x_new
            fx = fx_new
            # Adjust h for next iteration
            h = max(1e-12, 1e-8 * max(1.0, abs(x)))

    return best_x
