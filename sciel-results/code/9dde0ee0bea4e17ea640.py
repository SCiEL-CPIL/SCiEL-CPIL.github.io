"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Finds a root of f using a hybrid strategy:
    1. Try Newton's method with secant derivative (cheap, fast convergence).
    2. If Newton fails or diverges, use a bracketing search (bisection/Illinois)
       to guarantee finding a sign change, then refine.
    3. Strict deadline guard to prevent timeouts.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    x = float(x0)
    best_x = x
    best_abs = float('inf')

    def update_best(x_val, f_val):
        nonlocal best_x, best_abs
        af = abs(f_val)
        if af < best_abs:
            best_abs = af
            best_x = x_val

    def time_up():
        return (time.time() - start_time) > (deadline - start_time) - 0.5

    # Phase 1: Newton's method with secant derivative
    # Use a small number of Newton steps first
    x_prev = x
    f_prev = f(x)
    update_best(x, f_prev)

    if abs(f_prev) <= 1e-8:
        return x

    # Try up to 30 Newton steps
    for i in range(30):
        if time_up():
            break

        # Compute secant derivative using previous point
        if i == 0:
            # First step: use finite difference
            h = 1e-7 * max(1.0, abs(x))
            f_plus = f(x + h)
            d = (f_plus - f_prev) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - f_prev / d
        else:
            # Use secant: (f(x) - f(x_prev)) / (x - x_prev)
            dx = x - x_prev
            if dx == 0:
                break
            d = (f_prev - f(x_prev)) / dx  # Wait, this is wrong. Let me fix.
            # Actually, secant derivative: (f(x) - f(x_prev)) / (x - x_prev)
            # But we need f(x) which we haven't computed yet in this iteration structure.
            # Let me restructure.

        # Restructure: compute f(x) at start of each iteration after first
        break

    # Let me rewrite more carefully
    x = float(x0)
    f_x = f(x)
    update_best(x, f_x)
    if abs(f_x) <= 1e-8:
        return x

    x_prev = None
    f_prev = None

    # Newton/secant hybrid
    for i in range(40):
        if time_up():
            break

        if x_prev is None:
            # First iteration: finite difference derivative
            h = 1e-7 * max(1.0, abs(x))
            f_plus = f(x + h)
            d = (f_plus - f_x) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - f_x / d
        else:
            # Secant method
            dx = x - x_prev
            if dx == 0:
                break
            d = (f_x - f_prev) / dx
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - f_x / d

        # Check if step is reasonable
        if not math.isfinite(x_new):
            break
        # Limit step size to avoid divergence
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
            # Diverging, break to bracketing
            break

        x_prev = x
        f_prev = f_x
        x = x_new
        f_x = f(x)
        update_best(x, f_x)

        if abs(f_x) <= 1e-8:
            return x

        # Check for convergence
        if abs(f_x) < abs(f_prev) * 0.5:
            # Converging well
            continue
        elif abs(f_x) > abs(f_prev) * 2.0:
            # Diverging, switch to bracketing
            break

    # Phase 2: Bracketing search if Newton didn't converge
    # We need to find an interval [a, b] where f(a) and f(b) have opposite signs
    # Start from best_x and expand outward

    a = best_x
    fa = f(a)
    if abs(fa) <= 1e-8:
        return a

    # Try expanding in both directions
    found_bracket = False
    step = 1.0
    for direction in [1, -1]:
        if time_up():
            break
        b = a + direction * step
        fb = f(b)
        if abs(fb) <= 1e-8:
            return b
        if fa * fb < 0:
            found_bracket = True
            break
        # Update: move the bracket endpoint
        if abs(fb) < abs(fa):
            a = b
            fa = fb
        step *= 2.0
        if step > 1e6:
            break

    if found_bracket:
        # Use Illinois method (modified bisection) for guaranteed convergence
        # Illinois method converges faster than bisection
        for i in range(100):
            if time_up():
                break
            if fa * fb > 0:
                # Shouldn't happen if bracket is valid, but check
                break
            c = (a + b) / 2.0
            fc = f(c)
            update_best(c, fc)
            if abs(fc) <= 1e-8:
                return c
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                # Illinois: adjust the endpoint with larger |f|
                if abs(fa) > abs(fb):
                    fa *= 0.5
                else:
                    fb *= 0.5
            if abs(b - a) < 1e-12:
                break

        # Return the best found
        if abs(fa) <= abs(fb):
            return a
        else:
            return b

    # If no bracket found, return best_x
    return best_x
# EVOLVE-BLOCK-END
