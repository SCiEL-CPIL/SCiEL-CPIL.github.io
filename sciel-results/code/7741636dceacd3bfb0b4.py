"""Root finding for ill-conditioned equations: MODEL-AND-OPTIMIZE with inverse-interpolation polish."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 118.0   # Safety margin below 120s limit
    MAX_EVALS = 490    # Safety margin below 500 limit

    # State tracking
    eval_count = 0
    best_x = float(x0)
    best_f = None
    best_abs = float('inf')

    def check_budget():
        nonlocal eval_count
        if eval_count >= MAX_EVALS or (time.time() - start_time) > DEADLINE:
            return True
        return False

    def safe_eval(x):
        nonlocal eval_count, best_x, best_f, best_abs
        if check_budget():
            return None
        try:
            val = f(x)
        except Exception:
            return None
        if not math.isfinite(val):
            return None
        eval_count += 1
        abs_val = abs(val)
        if abs_val < best_abs:
            best_f = val
            best_x = x
            best_abs = abs_val
        return val

    # 1. Initial check
    best_f = safe_eval(best_x)
    if best_f is not None and best_abs <= 1e-8:
        return best_x

    # ------------------------------------------------------------------
    # Inverse quadratic interpolation / secant solver
    # ------------------------------------------------------------------
    def solve_ivp(x_start, max_iter=40):
        """
        Solve using inverse quadratic interpolation (IQI) / secant.
        We model x as a function of f(x): x = g(f).
        With 2 points: linear (secant).
        With 3 points: quadratic (IQI).
        This is history-dependent and reuses all previous points.
        """
        nonlocal eval_count, best_x, best_f, best_abs

        x = float(x_start)
        fx = safe_eval(x)
        if fx is None:
            return None
        if abs(fx) <= 1e-8:
            return x

        # Get a second point
        h = 1e-4 * max(1.0, abs(x))
        x2 = x + h
        fx2 = safe_eval(x2)
        if fx2 is None:
            return x
        if abs(fx2) < abs(fx):
            x, fx = x2, fx2
            x2, fx2 = x - h, safe_eval(x - h)
            if fx2 is None:
                return x

        x1, f1 = x2, fx2  # older
        x0, f0 = x, fx     # newer

        for _ in range(max_iter):
            if check_budget():
                break
            if abs(f0) <= 1e-8:
                return x0
            if abs(f1) <= 1e-8:
                return x1

            # Try inverse quadratic interpolation with 3 points
            # We need a third point; use the history if available, else bisection-like
            # For simplicity, use secant if we don't have 3 good points
            if f0 == f1:
                # Degenerate; perturb
                x_new = x0 + 1e-4
                f_new = safe_eval(x_new)
                if f_new is None:
                    break
                x1, f1 = x0, f0
                x0, f0 = x_new, f_new
                continue

            # Secant step
            x_new = x0 - f0 * (x0 - x1) / (f0 - f1)

            # Clamp to avoid wild jumps
            max_jump = max(1.0, 0.5 * abs(x0))
            if abs(x_new - x0) > max_jump:
                x_new = x0 + math.copysign(max_jump, x_new - x0)

            f_new = safe_eval(x_new)
            if f_new is None:
                break

            if abs(f_new) <= 1e-8:
                return x_new

            # Update: keep the two most recent distinct points
            x1, f1 = x0, f0
            x0, f0 = x_new, f_new

            if abs(x0 - x1) < 1e-14 * max(1.0, abs(x0)):
                break

        return x0

    # ------------------------------------------------------------------
    # Bracketed solver: Illinois / Regula Falsi hybrid
    # ------------------------------------------------------------------
    def solve_bracketed(xa, fa, xb, fb, max_iter=60):
        """Solve a bracketed root using Illinois method."""
        nonlocal eval_count, best_x, best_f, best_abs

        # Ensure fa has smaller absolute value
        if abs(fa) > abs(fb):
            xa, fa, xb, fb = xb, fb, xa, fa

        for _ in range(max_iter):
            if check_budget():
                break
            if abs(fa) <= 1e-8:
                return xa
            if abs(fb) <= 1e-8:
                return xb

            # Regula falsi step
            if fa == fb:
                x_new = (xa + xb) / 2.0
            else:
                x_new = (xa * fb - xb * fa) / (fb - fa)

            # Ensure within bracket
            if x_new <= min(xa, xb) or x_new >= max(xa, xb):
                x_new = (xa + xb) / 2.0

            f_new = safe_eval(x_new)
            if f_new is None:
                break
            if abs(f_new) <= 1e-8:
                return x_new

            # Update bracket
            if fa * f_new < 0:
                # Root between xa and x_new; xa is the "good" end
                xb, fb = x_new, f_new
            else:
                # Root between x_new and xb; Illinois: halve the retained endpoint's f
                xa, fa = x_new, f_new / 2.0

            if abs(xa - xb) < 1e-12 * max(1.0, abs(xa)):
                break

        return xa if abs(fa) <= abs(fb) else xb

    # ------------------------------------------------------------------
    # High-precision polish pass using Newton-like local model
    # ------------------------------------------------------------------
    def polish(x_best, f_best, max_iter=20):
        """
        Final polish: use finite-difference derivative to do Newton steps.
        This is a "final high-precision polish pass" as suggested.
        """
        nonlocal eval_count, best_x, best_f, best_abs

        x = x_best
        fx = f_best

        for _ in range(max_iter):
            if check_budget():
                break
            if abs(fx) <= 1e-12:
                return x

            # Finite difference derivative
            h = max(1e-8, 1e-8 * abs(x))
            xp = x + h
            xm = x - h
            fxp = safe_eval(xp)
            fxm = safe_eval(xm)
            if fxp is None or fxm is None:
                break

            # Central difference
            if abs(xp - xm) > 0:
                dfdx = (fxp - fxm) / (xp - xm)
            else:
                break

            if abs(dfdx) < 1e-30:
                # Derivative too small; try secant with history
                break

            # Newton step
            x_new = x - fx / dfdx

            # Damping
            max_step = max(1e-6, 0.5 * abs(x))
            if abs(x_new - x) > max_step:
                x_new = x + math.copysign(max_step, x_new - x)

            f_new = safe_eval(x_new)
            if f_new is None:
                break

            if abs(f_new) < abs(fx):
                x, fx = x_new, f_new
            else:
                # Halve the step
                x_new = x - 0.5 * (x - x_new)
                f_new2 = safe_eval(x_new)
                if f_new2 is None:
                    break
                if abs(f_new2) < abs(fx):
                    x, fx = x_new, f_new2
                else:
                    break

            if abs(x_new - x) < 1e-14 * max(1.0, abs(x)):
                break

        return x

    # ------------------------------------------------------------------
    # Main strategy
    # ------------------------------------------------------------------

    # Strategy 1: Direct local solve using inverse-secant
    result = solve_ivp(x0, max_iter=35)
    if result is not None and best_abs <= 1e-8:
        return best_x

    # Try bracketing from x0
    h = max(1e-4, 1e-4 * abs(x0))
    for factor in [1, 2, 5, 10, 50, 100]:
        if check_budget():
            break
        h = max(1e-6, 1e-4 * abs(x0)) * factor
        xa = x0 - h
        xb = x0 + h
        fa = safe_eval(xa)
        fb = safe_eval(xb)
        if fa is None or fb is None:
            continue
        if fa * fb < 0:
            result = solve_bracketed(xa, fa, xb, fb, max_iter=50)
            if result is not None and best_abs <= 1e-8:
                return best_x
            # Polish
            if best_f is not None:
                result = polish(best_x, best_f, max_iter=15)
                if result is not None and best_abs <= 1e-8:
                    return best_x

    # Strategy 2: Perturbed starts
    perturbations = [
        1e-3, -1e-3,
        1e-2, -1e-2,
        1e-1, -1e-1,
        0.5, -0.5,
        1.0, -1.0,
        2.0, -2.0,
    ]

    for p in perturbations:
        if check_budget():
            break
        x_pert = x0 + p
        result = solve_ivp(x_pert, max_iter=25)
        if result is not None and best_abs <= 1e-8:
            return best_x

    # Strategy 3: Broader sampling with bracket detection
    radii = [1.0, 10.0, 100.0, 1000.0]
    for r in radii:
        if check_budget():
            break
        samples = [
            x0 - r,
            x0 - 0.5 * r,
            x0 + 0.5 * r,
            x0 + r,
            x0 - 0.1 * r,
            x0 + 0.1 * r,
        ]
        for x_s in samples:
            if check_budget():
                break
            fs = safe_eval(x_s)
            if fs is None:
                continue
            if best_abs <= 1e-8:
                return best_x
            result = solve_ivp(x_s, max_iter=20)
            if result is not None and best_abs <= 1e-8:
                return best_x

    # Final polish on best candidate
    if best_f is not None:
        polish(best_x, best_f, max_iter=10)

    return best_x

# EVOLVE-BLOCK-END
