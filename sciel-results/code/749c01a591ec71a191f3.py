"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Combines multi-start random restarts with safeguarded inverse quadratic
    interpolation (SIQI) and derivative-free minimization fallbacks.
    """
    x0 = float(x0)
    deadline = time.monotonic() + 138.0  # safety margin before 150s kill
    max_evals = 490
    eval_count = [0]

    best_x = x0
    best_f_val = None
    best_abs = math.inf

    def eval_f(x):
        nonlocal best_x, best_f_val, best_abs
        if time.monotonic() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = float(f(x))
            eval_count[0] += 1
            if not math.isfinite(val):
                return None
            av = abs(val)
            if av < best_abs:
                best_abs = av
                best_x = float(x)
                best_f_val = val
            return val
        except (ValueError, OverflowError, ZeroDivisionError, TypeError, AttributeError, ArithmeticError):
            eval_count[0] += 1
            return None

    fx0 = eval_f(x0)
    if fx0 is None:
        return best_x
    if abs(fx0) <= 1e-8:
        return best_x

    def siqi_solve(a, b, fa, fb, max_iter=60):
        """
        Safeguarded Inverse Quadratic Interpolation (SIQI) variant.
        Maintains a bracket [a, b] with sign change and uses secant-like
        steps safeguarded by bisection.
        """
        if fa * fb > 0:
            return None

        # Ensure |fa| < |fb| for standard SIQI formulation
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        c = a
        fc = fa
        d = b - a
        e = d

        for _ in range(max_iter):
            if time.monotonic() > deadline or eval_count[0] >= max_evals - 2:
                break

            if abs(fa) <= 1e-12:
                return a
            if abs(fb) <= 1e-12:
                return b

            # If bracket is very small, return best endpoint
            if abs(d) < 1e-14 * max(1.0, abs(a)):
                return a if abs(fa) < abs(fb) else b

            x_try = None

            # Try secant step
            denom = fb - fa
            if abs(denom) > 1e-300:
                x_secant = b - fb * (b - a) / denom
                mid = (a + b) / 2.0
                # Safeguard: accept secant if it's inside bracket and not too close to mid
                if a < x_secant < b:
                    if abs(x_secant - mid) > 0.1 * abs(b - a):
                        x_try = x_secant
                    else:
                        x_try = mid
                else:
                    x_try = mid
            else:
                x_try = (a + b) / 2.0

            fx_try = eval_f(x_try)
            if fx_try is None:
                break

            if abs(fx_try) <= 1e-8:
                return x_try

            # Update bracket
            if fa * fx_try < 0:
                b = x_try
                fb = fx_try
                d = b - a
                e = d
            else:
                c = a
                fc = fa
                a = x_try
                fa = fx_try
                d = c - a
                e = b - a

        # Return best endpoint
        return a if abs(fa) < abs(fb) else b

    def try_bracket_and_solve(x_start, fx_start, scales):
        """
        Try to find a bracket around x_start using given scales, then solve with SIQI.
        Returns True if a solution is found.
        """
        if fx_start is None:
            fx_start = eval_f(x_start)
            if fx_start is None:
                return False
            if abs(fx_start) <= 1e-8:
                return True

        for s in scales:
            if time.monotonic() > deadline or eval_count[0] >= max_evals - 5:
                break

            a = x_start - s
            b = x_start + s

            fa = eval_f(a)
            if fa is None:
                continue
            fb = eval_f(b)
            if fb is None:
                continue

            # Check for sign change
            if fa * fb < 0:
                root = siqi_solve(a, b, fa, fb)
                if root is not None:
                    f_root = eval_f(root)
                    if f_root is not None and abs(f_root) <= 1e-8:
                        return True
                    # Even if not perfect, update best
                    if f_root is not None and abs(f_root) < best_abs:
                        pass  # eval_f already updates best
                break  # Found a bracket, no need to try larger scales

        return False

    # Phase 1: Local expansion from x0
    scales = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    if try_bracket_and_solve(x0, fx0, scales):
        return best_x

    # Phase 2: Newton's method with finite difference derivative
    def try_newton(x_start, fx_start, max_iter=30):
        x = x_start
        fx = fx_start
        for i in range(max_iter):
            if time.monotonic() > deadline or eval_count[0] >= max_evals - 2:
                break
            if abs(fx) <= 1e-8:
                return x

            h = 1e-7 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            if fx_plus is None:
                break
            d = (fx_plus - fx) / h
            if d == 0 or not math.isfinite(d):
                break

            step = fx / d
            # Damp the step
            max_step = abs(x) * 0.5 + 1.0
            if abs(step) > max_step:
                step = math.copysign(max_step, step)

            x_new = x - step
            fx_new = eval_f(x_new)
            if fx_new is None:
                break

            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                # Try smaller step
                step *= 0.5
                x_new = x - step
                fx_new = eval_f(x_new)
                if fx_new is None:
                    break
                if abs(fx_new) < abs(fx):
                    x, fx = x_new, fx_new
                else:
                    break
        return x if abs(fx) <= 1e-8 else None

    newton_result = try_newton(x0, fx0)
    if newton_result is not None:
        return newton_result

    # Phase 3: Multi-start random restarts
    rng = np.random.RandomState(42)
    radii = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    for radius in radii:
        if time.monotonic() > deadline or eval_count[0] >= max_evals - 10:
            break
        for _ in range(5):
            if time.monotonic() > deadline or eval_count[0] >= max_evals - 10:
                break
            angle = rng.uniform(0, 2 * math.pi)
            x_try = x0 + radius * math.cos(angle)
            fx_try = eval_f(x_try)
            if fx_try is None:
                continue
            if abs(fx_try) <= 1e-8:
                return best_x

            # Try bracket + SIQI
            if try_bracket_and_solve(x_try, fx_try, [1e-3, 1e-2, 0.1, 1.0, 5.0]):
                return best_x

            # Try Newton
            nr = try_newton(x_try, fx_try, max_iter=15)
            if nr is not None:
                return nr

    # Phase 4: Secant method with perturbed points
    h = 1e-5 * max(1.0, abs(x0))
    x2 = x0 + h
    fx2 = eval_f(x2)
    if fx2 is not None:
        # Secant iteration
        x1, fx1 = x0, fx0
        for _ in range(20):
            if time.monotonic() > deadline or eval_count[0] >= max_evals - 2:
                break
            if abs(fx1) <= 1e-8:
                return x1
            if abs(fx2) <= 1e-8:
                return x2
            if fx1 == fx2:
                break
            x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            x1, fx1 = x2, fx2
            x2, fx2 = x_new, fx_new

    return best_x
# EVOLVE-BLOCK-END
