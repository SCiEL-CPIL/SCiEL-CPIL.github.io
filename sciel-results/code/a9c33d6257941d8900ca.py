"""Root finding: Ill-conditioned - robust solver with multiple strategies."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a portfolio approach: Newton with adaptive step, secant, bisection,
    and Illinois method for ill-conditioned problems.
    """
    start_time = time.time()
    MAX_WALL = 115.0  # Safety margin before 120s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8

    # Global evaluation counter
    eval_count = [0]

    def eval_f(x):
        """Counted evaluation of f."""
        if eval_count[0] >= MAX_EVALS:
            return 0.0
        eval_count[0] += 1
        val = f(x)
        if not math.isfinite(val):
            return 0.0
        return val

    def is_good(x, fx):
        """Check if we have a good enough solution."""
        return abs(fx) <= TARGET_TOL

    def check_time():
        """Check if we've exceeded the time limit."""
        return (time.time() - start_time) > MAX_WALL

    def newton_with_adaptive_step(x, fx):
        """
        Newton's method with adaptive step sizing and progress monitoring.
        Returns (x, fx, success) where success indicates convergence.
        """
        x = float(x)
        fx = float(fx)
        if is_good(x, fx):
            return x, fx, True

        best_x, best_fx = x, fx
        best_abs = abs(fx)
        h_base = 1e-7 * max(1.0, abs(x))
        prev_abs = best_abs
        stall_count = 0

        for _ in range(40):
            if check_time() or eval_count[0] >= MAX_EVALS:
                break

            # Compute derivative using central difference
            h = h_base
            fx_plus = eval_f(x + h)
            fx_minus = eval_f(x - h)
            d = (fx_plus - fx_minus) / (2.0 * h)

            if d == 0 or not math.isfinite(d):
                break

            # Newton step
            step = fx / d
            x_new = x - step

            # Adaptive step sizing: limit step to avoid divergence
            max_step = 0.5 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = math.copysign(max_step, step)
                x_new = x - step

            fx_new = eval_f(x_new)
            if not math.isfinite(fx_new):
                # Try smaller step
                step *= 0.5
                x_new = x - step
                fx_new = eval_f(x_new)
                if not math.isfinite(fx_new):
                    break

            abs_new = abs(fx_new)
            if abs_new < best_abs:
                best_x, best_fx = x_new, fx_new
                best_abs = abs_new

            # Check for convergence
            if abs_new <= TARGET_TOL:
                return x_new, fx_new, True

            # Monitor progress
            if abs_new < prev_abs * 0.9:
                stall_count = 0
            else:
                stall_count += 1

            prev_abs = abs_new

            # Damp the step if we're stalling
            if stall_count >= 3:
                step = fx / d * 0.5
                x_new = x - step
                fx_new = eval_f(x_new)
                if math.isfinite(fx_new) and abs(fx_new) < best_abs:
                    best_x, best_fx = x_new, fx_new
                    best_abs = abs(fx_new)
                stall_count = 0

            x, fx = x_new, fx_new

            if is_good(x, fx):
                return x, fx, True

        return best_x, best_fx, False

    def secant_method(x0, x1, fx0, fx1):
        """
        Secant method with safeguards.
        Returns (x, fx, success).
        """
        x, fx = x0, fx0
        x1, fx1 = x1, fx1

        # Ensure we have two distinct points
        if x1 == x:
            x1 = x + 1e-6
            fx1 = eval_f(x1)

        best_x, best_fx = min((x, x1), key=lambda v: abs(fx if v == x else fx1))
        best_abs = min(abs(fx), abs(fx1))
        if best_x == x:
            best_fx = fx
        else:
            best_fx = fx1

        prev_abs = best_abs
        stall_count = 0

        for _ in range(40):
            if check_time() or eval_count[0] >= MAX_EVALS:
                break

            d = (fx1 - fx) / (x1 - x) if x1 != x else 0.0
            if d == 0 or not math.isfinite(d):
                break

            x_new = x1 - fx1 / d

            # Safeguard: keep x_new within a reasonable range
            x_min = min(x, x1) - max(1.0, abs(x1))
            x_max = max(x, x1) + max(1.0, abs(x1))
            if x_new < x_min:
                x_new = x_min
            elif x_new > x_max:
                x_new = x_max

            fx_new = eval_f(x_new)
            if not math.isfinite(fx_new):
                # Fall back to bisection-like step
                x_new = (x + x1) / 2
                fx_new = eval_f(x_new)
                if not math.isfinite(fx_new):
                    break

            abs_new = abs(fx_new)
            if abs_new < best_abs:
                best_x, best_fx = x_new, fx_new
                best_abs = abs_new

            if abs_new <= TARGET_TOL:
                return x_new, fx_new, True

            # Monitor progress
            if abs_new < prev_abs * 0.9:
                stall_count = 0
            else:
                stall_count += 1

            prev_abs = abs_new

            # Shift points
            x, fx = x1, fx1
            x1, fx1 = x_new, fx_new

            if stall_count >= 5:
                break

        return best_x, best_fx, False

    def bisection(a, b, fa, fb):
        """
        Standard bisection method. Requires fa and fb to have opposite signs.
        Returns (x, fx, success).
        """
        if fa * fb > 0:
            return a, fa, False  # No sign change

        best_x, best_fx = (a, fa) if abs(fa) < abs(fb) else (b, fb)
        best_abs = min(abs(fa), abs(fb))

        for _ in range(50):
            if check_time() or eval_count[0] >= MAX_EVALS:
                break

            mid = (a + b) / 2
            fm = eval_f(mid)
            if not math.isfinite(fm):
                break

            abs_m = abs(fm)
            if abs_m < best_abs:
                best_x, best_fx = mid, fm
                best_abs = abs_m

            if abs_m <= TARGET_TOL:
                return mid, fm, True

            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm

            # Check for convergence based on interval width
            if abs(b - a) < 1e-15 * max(1.0, abs(mid)):
                break

        return best_x, best_fx, False

    def illinois_method(a, b, fa, fb):
        """
        Illinois method: modified secant that shrinks the larger bracket
        sub-interval by a fixed factor (0.5) instead of bisection.
        Specifically designed for ill-conditioned functions near roots.
        Returns (x, fx, success).
        """
        if fa * fb > 0:
            return a, fa, False  # No sign change

        best_x, best_fx = (a, fa) if abs(fa) < abs(fb) else (b, fb)
        best_abs = min(abs(fa), abs(fb))

        # Illinois method: maintain a bracket [a, b] with f(a)*f(b) <= 0
        # At each step, compute the secant root, then shrink the larger
        # sub-interval by a factor of 0.5 (instead of replacing the endpoint
        # with the secant root as in standard Illinois).
        for _ in range(50):
            if check_time() or eval_count[0] >= MAX_EVALS:
                break

            # Secant estimate
            if fb != fa:
                x_sec = (a * fb - b * fa) / (fb - fa)
            else:
                x_sec = (a + b) / 2

            fx_sec = eval_f(x_sec)
            if not math.isfinite(fx_sec):
                # Fall back to bisection
                x_sec = (a + b) / 2
                fx_sec = eval_f(x_sec)
                if not math.isfinite(fx_sec):
                    break

            abs_sec = abs(fx_sec)
            if abs_sec < best_abs:
                best_x, best_fx = x_sec, fx_sec
                best_abs = abs_sec

            if abs_sec <= TARGET_TOL:
                return x_sec, fx_sec, True

            # Update bracket
            if fa * fx_sec <= 0:
                # Root is in [a, x_sec]
                b = x_sec
                fb = fx_sec
            else:
                # Root is in [x_sec, b]
                a = x_sec
                fa = fx_sec

            # Illinois modification: shrink the larger sub-interval by 0.5
            # Instead of just replacing the endpoint, we pull it inward
            # toward the secant point to accelerate convergence on
            # ill-conditioned problems.
            interval = abs(b - a)
            if interval < 1e-15 * max(1.0, abs((a + b) / 2)):
                break

            # Shrink the side that was just updated (the larger side)
            # by a factor of 0.5 toward the interior
            mid = (a + b) / 2
            if (b - x_sec) > (x_sec - a):
                # The right side is larger; shrink b toward mid
                b = x_sec + 0.5 * (b - x_sec)
                fb = eval_f(b)
                if not math.isfinite(fb):
                    b = (a + b) / 2
                    fb = eval_f(b)
            else:
                # The left side is larger; shrink a toward mid
                a = x_sec + 0.5 * (a - x_sec)
                fa = eval_f(a)
                if not math.isfinite(fa):
                    a = (a + b) / 2
                    fa = eval_f(a)

            abs_a = abs(fa)
            abs_b = abs(fb)
            if abs_a < best_abs:
                best_x, best_fx = a, fa
                best_abs = abs_a
            if abs_b < best_abs:
                best_x, best_fx = b, fb
                best_abs = abs_b

            if best_abs <= TARGET_TOL:
                return best_x, best_fx, True

        return best_x, best_fx, False

    def find_bracket(x0, fx0):
        """
        Try to find a sign-change bracket around x0 using adaptive step-doubling.
        Returns (a, b, fa, fb) or None if no bracket found.
        """
        if fx0 == 0:
            return None

        # Try expanding in both directions
        step = max(1e-3, 1e-3 * abs(x0))
        a, b = None, None
        fa, fb = None, None

        # Expand right
        for i in range(20):
            if check_time() or eval_count[0] >= MAX_EVALS:
                break
            x_right = x0 + step
            fx_right = eval_f(x_right)
            if not math.isfinite(fx_right):
                break
            if fx0 * fx_right <= 0:
                a, fa = x0, fx0
                b, fb = x_right, fx_right
                return a, b, fa, fb
            step *= 2

        # Expand left
        step = max(1e-3, 1e-3 * abs(x0))
        for i in range(20):
            if check_time() or eval_count[0] >= MAX_EVALS:
                break
            x_left = x0 - step
            fx_left = eval_f(x_left)
            if not math.isfinite(fx_left):
                break
            if fx0 * fx_left <= 0:
                a, fa = x_left, fx_left
                b, fb = x0, fx0
                return a, b, fa, fb
            step *= 2

        return None

    # --- Main solver ---
    x0 = float(x0)
    fx0 = eval_f(x0)

    if is_good(x0, fx0):
        return x0

    best_x, best_fx = x0, fx0
    best_abs = abs(fx0)

    def update_best(x, fx):
        nonlocal best_x, best_fx, best_abs
        if math.isfinite(fx) and abs(fx) < best_abs:
            best_x, best_fx = x, fx
            best_abs = abs(fx)

    # Strategy 1: Try to find a bracket and use Illinois/bisection
    bracket = find_bracket(x0, fx0)
    if bracket is not None:
        a, b, fa, fb = bracket
        update_best(a, fa)
        update_best(b, fb)

        # Try Illinois method first (designed for ill-conditioned)
        x_ill, fx_ill, success_ill = illinois_method(a, b, fa, fb)
        update_best(x_ill, fx_ill)
        if success_ill:
            return x_ill

        # Try bisection as fallback
        x_bis, fx_bis, success_bis = bisection(a, b, fa, fb)
        update_best(x_bis, fx_bis)
        if success_bis:
            return x_bis

    # Strategy 2: Newton with adaptive step
    x_new, fx_new, success_new = newton_with_adaptive_step(x0, fx0)
    update_best(x_new, fx_new)
    if success_new:
        return x_new

    # Strategy 3: Secant method
    # Need a second point; use a small perturbation
    x1 = x0 + 1e-6 * max(1.0, abs(x0))
    fx1 = eval_f(x1)
    if math.isfinite(fx1):
        x_sec, fx_sec, success_sec = secant_method(x0, x1, fx0, fx1)
        update_best(x_sec, fx_sec)
        if success_sec:
            return x_sec

    # Strategy 4: Try multiple restarts with different initial points
    offsets = [1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]
    for off in offsets:
        if check_time() or eval_count[0] >= MAX_EVALS:
            break
        x_try = x0 + off * max(1.0, abs(x0))
        fx_try = eval_f(x_try)
        if not math.isfinite(fx_try):
            continue
        update_best(x_try, fx_try)

        if is_good(x_try, fx_try):
            return x_try

        # Try Newton from this point
        x_n, fx_n, s_n = newton_with_adaptive_step(x_try, fx_try)
        update_best(x_n, fx_n)
        if s_n:
            return x_n

        # Try to bracket from this point
        br = find_bracket(x_try, fx_try)
        if br is not None:
            a2, b2, fa2, fb2 = br
            update_best(a2, fa2)
            update_best(b2, fb2)
            x_il, fx_il, s_il = illinois_method(a2, b2, fa2, fb2)
            update_best(x_il, fx_il)
            if s_il:
                return x_il

    # Final polish: try a few more secant/Newton iterations from best point
    for _ in range(10):
        if check_time() or eval_count[0] >= MAX_EVALS:
            break
        if is_good(best_x, best_fx):
            return best_x

        # Try Newton
        x_n, fx_n, s_n = newton_with_adaptive_step(best_x, best_fx)
        update_best(x_n, fx_n)
        if s_n:
            return x_n

        # Try secant with a nearby point
        x_near = best_x + 1e-8 * max(1.0, abs(best_x))
        fx_near = eval_f(x_near)
        if math.isfinite(fx_near) and fx_near != best_fx:
            x_s, fx_s, s_s = secant_method(best_x, x_near, best_fx, fx_near)
            update_best(x_s, fx_s)
            if s_s:
                return x_s

    return best_x

# EVOLVE-BLOCK-END
