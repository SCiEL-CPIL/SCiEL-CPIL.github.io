"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, using a robust portfolio approach.
    Prioritizes bracketing + Brent-like method, with Newton/secant fallbacks.
    Uses deadline guard to stay within wall-clock budget.
    """
    deadline = time.time() + 145.0
    tol = 1e-9  # target |f(x)| <= 1e-8, use tighter internal tol

    x = float(x0)

    # Best solution found so far
    best_x = x
    best_f = None
    best_abs = float('inf')

    def update_best(x_val, f_val):
        nonlocal best_x, best_f, best_abs
        if f_val is None or not math.isfinite(x_val) or not math.isfinite(f_val):
            return
        abs_f = abs(f_val)
        if abs_f < best_abs:
            best_x = x_val
            best_f = f_val
            best_abs = abs_f

    def check_deadline():
        return time.time() >= deadline

    # Initial evaluation
    try:
        f_x0 = f(x)
        if f_x0 is not None and math.isfinite(f_x0):
            update_best(x, f_x0)
            if abs(f_x0) <= tol:
                return x
        else:
            f_x0 = None
    except:
        f_x0 = None

    # Phase 1: Try to find a bracket around x0
    # This is the most reliable method for oscillatory functions
    a = x
    fa = f_x0
    b = None
    fb = None

    if fa is not None:
        # Try expanding in both directions
        step = 1e-4 * max(1.0, abs(x))
        max_expansions = 30

        for i in range(max_expansions):
            if check_deadline():
                break

            current_step = step * (2 ** i)

            # Try positive direction
            x_pos = x + current_step
            try:
                f_pos = f(x_pos)
                if f_pos is not None and math.isfinite(f_pos):
                    update_best(x_pos, f_pos)
                    if fa * f_pos < 0:
                        b = x_pos
                        fb = f_pos
                        break
            except:
                pass

            # Try negative direction
            x_neg = x - current_step
            try:
                f_neg = f(x_neg)
                if f_neg is not None and math.isfinite(f_neg):
                    update_best(x_neg, f_neg)
                    if fa * f_neg < 0:
                        a = x_neg
                        b = x
                        fa = f_neg
                        fb = f_x0
                        break
            except:
                pass

            # If step gets too large, stop
            if current_step > 1e8:
                break

    # Phase 2: If bracket found, use Brent-like method (secant + bisection hybrid)
    if b is not None and fa is not None and fb is not None and fa * fb < 0:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Brent-like iteration: try secant, fall back to bisection
        c = a
        fc = fa
        d = 0.0
        e = 0.0

        for _ in range(200):
            if check_deadline():
                break

            # If f(a) is close enough, return
            if abs(fa) <= tol:
                return a
            if abs(fb) <= tol:
                return b

            # Check for convergence
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                break

            # Try secant step
            secant_ok = False
            x_new = None

            if fa != fb:
                # Secant estimate
                x_sec = b - fb * (b - a) / (fb - fa)
                # Check if secant point is within bracket
                if a < x_sec < b:
                    # Additional check: secant step shouldn't be too small
                    if abs(x_sec - b) > 1e-14 * max(1.0, abs(b)):
                        x_new = x_sec
                        secant_ok = True

            # If secant not usable, use bisection
            if not secant_ok:
                x_new = (a + b) / 2.0

            # Evaluate f at x_new
            try:
                f_new = f(x_new)
                if f_new is None or not math.isfinite(f_new):
                    break
                update_best(x_new, f_new)
            except:
                break

            if abs(f_new) <= tol:
                return x_new

            # Update bracket
            if fa * f_new < 0:
                # Root is between a and x_new
                b = x_new
                fb = f_new
            else:
                # Root is between x_new and b
                a = x_new
                fa = f_new

    # Phase 3: If no bracket found, try Newton's method with finite-difference derivative
    if b is None:
        x_newton = x
        for _ in range(50):
            if check_deadline():
                break
            try:
                fx = f(x_newton)
                if fx is None or not math.isfinite(fx):
                    break
                update_best(x_newton, fx)
                if abs(fx) <= tol:
                    return x_newton
                h = 1e-6 * max(1.0, abs(x_newton))
                d = (f(x_newton + h) - fx) / h
                if d == 0 or not math.isfinite(d):
                    break
                x_newton = x_newton - fx / d
                if not math.isfinite(x_newton):
                    break
            except:
                break

    # Phase 4: Try secant method as another fallback
    if b is None:
        x0_sec = x
        x1_sec = x0_sec + max(1e-3, 1e-3 * abs(x0_sec))
        try:
            fx0 = f(x0_sec)
            fx1 = f(x1_sec)
        except:
            fx0 = None
            fx1 = None

        if fx0 is not None and fx1 is not None and math.isfinite(fx0) and math.isfinite(fx1):
            for _ in range(50):
                if check_deadline():
                    break
                if abs(fx0) <= tol:
                    return x0_sec
                if abs(fx1) <= tol:
                    return x1_sec
                if fx0 == fx1:
                    break
                x2_sec = x1_sec - fx1 * (x1_sec - x0_sec) / (fx1 - fx0)
                if not math.isfinite(x2_sec):
                    break
                try:
                    fx2 = f(x2_sec)
                    if fx2 is None or not math.isfinite(fx2):
                        break
                    update_best(x2_sec, fx2)
                except:
                    break
                x0_sec = x1_sec
                fx0 = fx1
                x1_sec = x2_sec
                fx1 = fx2

    # Return the best solution found
    if best_f is not None and best_abs <= 1e-8:
        return best_x
    return best_x
# EVOLVE-BLOCK-END
