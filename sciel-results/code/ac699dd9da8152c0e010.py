"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 145.0

    evals = 0
    best_x = float(x0)
    best_abs = float('inf')

    def call_f(x):
        nonlocal evals, best_x, best_abs
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return None
        if math.isnan(val) or math.isinf(val):
            return None
        a = abs(val)
        if a < best_abs:
            best_abs = a
            best_x = x
        return val

    def is_good(val):
        return val is not None and abs(val) <= TOL

    def newton_raphson(x0_val, max_iter=80):
        nonlocal evals, best_x, best_abs
        x = float(x0_val)
        fx = call_f(x)
        if fx is None:
            return best_x
        if is_good(fx):
            return x

        for _ in range(max_iter):
            if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
                break
            # Adaptive step size for derivative
            h = 1e-8 * max(1.0, abs(x))
            xp = x + h
            xm = x - h
            fpx = call_f(xp)
            fxm = call_f(xm)

            if fpx is not None and fxm is not None:
                d = (fpx - fxm) / (2 * h)
            elif fpx is not None:
                d = (fpx - fx) / h
            elif fxm is not None:
                d = (fx - fxm) / h
            else:
                break

            if d == 0 or not math.isfinite(d):
                break

            x_new = x - fx / d
            if not math.isfinite(x_new):
                break

            # Damping if step is too large
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x + math.copysign(10 * max(1.0, abs(x)), x_new - x)

            fx_new = call_f(x_new)
            if fx_new is None:
                break

            if is_good(fx_new):
                return x_new

            # Check for convergence
            if abs(x_new - x) < 1e-14 * max(1.0, abs(x_new)):
                break

            x = x_new
            fx = fx_new

        return best_x

    def secant(x0_val, max_iter=80):
        nonlocal evals, best_x, best_abs
        x0 = float(x0_val)
        x1 = x0 + 1e-4 * max(1.0, abs(x0))
        fx0 = call_f(x0)
        if fx0 is None:
            return best_x
        if is_good(fx0):
            return x0
        fx1 = call_f(x1)
        if fx1 is None:
            return best_x
        if is_good(fx1):
            return x1

        for _ in range(max_iter):
            if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
                break
            if fx1 == fx0:
                break
            x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            if not math.isfinite(x2):
                break
            # Damping
            if abs(x2 - x1) > 10 * max(1.0, abs(x1)):
                x2 = x1 + math.copysign(10 * max(1.0, abs(x1)), x2 - x1)
            x0, x1 = x1, x2
            fx0 = fx1
            fx1 = call_f(x1)
            if fx1 is None:
                break
            if is_good(fx1):
                return x1
        return x1

    # Try Newton first
    x_res = newton_raphson(x0)
    if is_good(call_f(x_res)):
        return x_res

    # Try Secant
    x_res = secant(x0)
    if is_good(call_f(x_res)):
        return x_res

    # Try Newton from a slightly perturbed start
    x_pert = x0 + 1e-3 * max(1.0, abs(x0))
    x_res = newton_raphson(x_pert)
    if is_good(call_f(x_res)):
        return x_res

    # Try Secant from perturbed start
    x_res = secant(x_pert)
    if is_good(call_f(x_res)):
        return x_res

    # Final high-precision polish pass: use best_x found so far, do a few Newton steps with tighter tolerance
    if best_abs > TOL and best_abs < 1e-4:
        x = best_x
        for _ in range(20):
            if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
                break
            fx = call_f(x)
            if fx is None:
                break
            if abs(fx) <= TOL:
                return x
            h = 1e-10 * max(1.0, abs(x))
            fpx = call_f(x + h)
            if fpx is None:
                break
            d = (fpx - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            fx_new = call_f(x_new)
            if fx_new is None:
                break
            if abs(fx_new) <= TOL:
                return x_new
            if abs(x_new - x) < 1e-15 * max(1.0, abs(x_new)):
                break
            x = x_new

    return best_x
# EVOLVE-BLOCK-END
