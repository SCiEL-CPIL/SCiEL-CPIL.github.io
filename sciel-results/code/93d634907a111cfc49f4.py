"""Root finder for 3 fixed oscillatory equations."""
import time
import math
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 138.0  # safety margin before 150s kill

    x0 = float(x0)

    def time_ok():
        return (time.time() - start_time) < TIME_LIMIT

    # Check initial point
    try:
        f0 = f(x0)
    except Exception:
        f0 = float('inf')

    if not np.isfinite(f0):
        f0 = float('inf')

    if abs(f0) <= 1e-8:
        return x0

    best_x = x0
    best_f = abs(f0)

    # Helper: safe function evaluation
    def safe_f(x):
        try:
            val = f(x)
            if not np.isfinite(val):
                return float('inf')
            return float(val)
        except Exception:
            return float('inf')

    def update_best(x, fx):
        nonlocal best_x, best_f
        if np.isfinite(fx) and abs(fx) < best_f:
            best_x = x
            best_f = abs(fx)

    # Helper: try to refine a sign-change bracket [a, b] using Brent's method
    def try_brent(a, b, fa, fb):
        if not (np.isfinite(fa) and np.isfinite(fb)):
            return None
        if fa * fb > 0:
            return None
        try:
            root = brentq(f, a, b, xtol=1e-14, rtol=1e-14, maxiter=200)
            fr = safe_f(root)
            if abs(fr) <= 1e-8:
                return root
        except Exception:
            pass
        # Fallback: bisection if Brent failed
        try:
            root = (a + b) / 2.0
            for _ in range(80):
                if not time_ok():
                    break
                root = (a + b) / 2.0
                fm = safe_f(root)
                update_best(root, fm)
                if abs(fm) <= 1e-10:
                    return root
                if fa * fm <= 0:
                    b = root
                    fb = fm
                else:
                    a = root
                    fa = fm
                if abs(b - a) < 1e-14 * max(1.0, abs(a)):
                    break
            root = (a + b) / 2.0
            if abs(safe_f(root)) <= 1e-8:
                return root
        except Exception:
            pass
        return None

    # Strategy 1: Logarithmic-magnitude sparse grid search
    # Evaluate f at x0 ± h for h in a geometric sequence to find the
    # coarsest scale where a sign change or near-zero value occurs.
    # This avoids linear call waste on functions with unknown periods.
    log_h = np.logspace(-8, 6, 60)  # h from 1e-8 to 1e6

    for h in log_h:
        if not time_ok():
            break

        for direction in [1, -1]:
            if not time_ok():
                break

            x_new = x0 + direction * h
            f_new = safe_f(x_new)

            if np.isfinite(f_new):
                update_best(x_new, f_new)

                # Check for sign change with x0
                if np.isfinite(f0) and f0 * f_new < 0:
                    root = try_brent(x0, x_new, f0, f_new)
                    if root is not None:
                        return root

                # Check for sign change with best_x if different
                if x_new != best_x and np.isfinite(safe_f(best_x)):
                    fb = safe_f(best_x)
                    if fb * f_new < 0:
                        root = try_brent(best_x, x_new, fb, f_new)
                        if root is not None:
                            return root

                # Check if f_new is very close to zero
                if abs(f_new) <= 1e-8:
                    return x_new

    # Strategy 2: Geometric expansion outward from x0
    # If the log-spaced sweep didn't find a bracket, expand outward
    # geometrically to find one.
    if time_ok() and best_f > 1e-8:
        step = 1e-3
        for _ in range(40):
            if not time_ok():
                break

            for direction in [1, -1]:
                if not time_ok():
                    break

                x_new = x0 + direction * step
                f_new = safe_f(x_new)

                if np.isfinite(f_new):
                    update_best(x_new, f_new)

                    if np.isfinite(f0) and f0 * f_new < 0:
                        root = try_brent(x0, x_new, f0, f_new)
                        if root is not None:
                            return root

                    if abs(f_new) <= 1e-8:
                        return x_new

            step *= 2.0

    # Strategy 3: Dense local sweep around best_x
    # If we still haven't found a bracket, do a denser sweep around
    # the best point found so far.
    if time_ok() and best_f > 1e-8:
        for scale in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            if not time_ok():
                break

            n_points = 30
            for direction in [1, -1]:
                if not time_ok():
                    break

                prev_x = best_x
                prev_f = safe_f(prev_x)

                for i in range(1, n_points + 1):
                    if not time_ok():
                        break

                    x_new = best_x + direction * i * scale
                    f_new = safe_f(x_new)

                    if np.isfinite(f_new):
                        update_best(x_new, f_new)

                        if np.isfinite(prev_f) and prev_f * f_new < 0:
                            root = try_brent(prev_x, x_new, prev_f, f_new)
                            if root is not None:
                                return root

                        if abs(f_new) <= 1e-8:
                            return x_new

                    prev_x = x_new
                    prev_f = f_new

    # Strategy 4: Newton's method from best point
    if time_ok() and best_f > 1e-8:
        x = best_x
        fx = safe_f(x)

        for _ in range(50):
            if not time_ok():
                break
            if abs(fx) <= 1e-10:
                return x

            # Numerical derivative
            h = 1e-7 * max(1.0, abs(x))
            fxh = safe_f(x + h)
            if not np.isfinite(fxh):
                break
            d = (fxh - fx) / h
            if d == 0 or not np.isfinite(d):
                break

            x_new = x - fx / d
            if not np.isfinite(x_new):
                break

            # Limit step size to avoid wild jumps
            max_step = 10.0 * max(1.0, abs(x))
            if abs(x_new - x) > max_step:
                x_new = x + max_step * (1 if x_new > x else -1)

            fx_new = safe_f(x_new)
            if abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
                update_best(x, fx)
            else:
                # Halve the step
                x_mid = (x + x_new) / 2.0
                fx_mid = safe_f(x_mid)
                if abs(fx_mid) < abs(fx):
                    x = x_mid
                    fx = fx_mid
                    update_best(x, fx)
                else:
                    break

    # Strategy 5: Secant method from best point
    if time_ok() and best_f > 1e-8:
        x1 = best_x
        f1 = safe_f(x1)
        dx = 1e-4 * max(1.0, abs(x1))
        x2 = x1 + dx
        f2 = safe_f(x2)

        if np.isfinite(f1) and np.isfinite(f2) and f1 != f2:
            for _ in range(50):
                if not time_ok():
                    break
                if abs(f1) <= 1e-10:
                    return x1
                if abs(f2) <= 1e-10:
                    return x2

                # Secant step
                x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
                if not np.isfinite(x_new):
                    break

                f_new = safe_f(x_new)
                if not np.isfinite(f_new):
                    break

                update_best(x_new, f_new)

                if abs(f_new) <= 1e-8:
                    return x_new

                # Update points
                x1, f1 = x2, f2
                x2, f2 = x_new, f_new

    # Strategy 6: Final high-precision polish pass
    # If we have a good candidate, try to polish it with tighter tolerance.
    if time_ok() and best_f > 1e-8:
        # Try a very fine local search around best_x
        for scale in [1e-6, 1e-7, 1e-8]:
            if not time_ok():
                break

            n_points = 20
            for direction in [1, -1]:
                if not time_ok():
                    break

                prev_x = best_x
                prev_f = safe_f(prev_x)

                for i in range(1, n_points + 1):
                    if not time_ok():
                        break

                    x_new = best_x + direction * i * scale
                    f_new = safe_f(x_new)

                    if np.isfinite(f_new):
                        update_best(x_new, f_new)

                        if np.isfinite(prev_f) and prev_f * f_new < 0:
                            root = try_brent(prev_x, x_new, prev_f, f_new)
                            if root is not None:
                                return root

                        if abs(f_new) <= 1e-8:
                            return x_new

                    prev_x = x_new
                    prev_f = f_new

    return best_x
# EVOLVE-BLOCK-END
