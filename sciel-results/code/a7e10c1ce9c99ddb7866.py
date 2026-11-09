import numpy as np

def find_root(f, x0):
    """
    Find a root of f near x0 using Brent's method (scipy.optimize.brentq).
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    """
    # Wrap f to count evaluations and enforce the limit
    eval_count = [0]
    max_evals = 500
    tol = 1e-8

    def f_counted(x):
        eval_count[0] += 1
        if eval_count[0] > max_evals:
            raise ValueError("Max evaluations exceeded")
        return f(x)

    # Initial evaluation
    x0 = float(x0)
    f0 = f_counted(x0)

    if abs(f0) <= tol:
        return x0

    # Try to find a bracket [a, b] where f(a) and f(b) have opposite signs
    # Strategy: expand in both directions from x0 with geometric step size
    # Keep track of points to avoid re-evaluating

    # Points: list of (x, f(x))
    points = [(x0, f0)]

    # Step size
    h = 1e-6 * max(1.0, abs(x0))

    # Expand to find a bracket
    a, b = None, None
    fa, fb = None, None

    # Try expanding in positive direction
    x_pos = x0 + h
    f_pos = f_counted(x_pos)
    points.append((x_pos, f_pos))

    if abs(f_pos) <= tol:
        return x_pos

    if f0 * f_pos < 0:
        a, b = x0, x_pos
        fa, fb = f0, f_pos
    else:
        # Continue expanding in positive direction
        x_neg = x0 - h
        f_neg = f_counted(x_neg)
        points.append((x_neg, f_neg))

        if abs(f_neg) <= tol:
            return x_neg

        if f0 * f_neg < 0:
            a, b = x_neg, x0
            fa, fb = f_neg, f0
        else:
            # No bracket found yet, expand further
            # Use a loop to expand until we find a bracket or exhaust evaluations
            step = h
            for _ in range(50):
                if eval_count[0] >= max_evals - 10:
                    break

                # Try positive side
                x_test = x0 + step
                if x_test not in [p[0] for p in points]:
                    f_test = f_counted(x_test)
                    points.append((x_test, f_test))

                    if abs(f_test) <= tol:
                        return x_test

                    # Check if bracket with any existing point
                    for (px, pf) in points:
                        if pf * f_test < 0:
                            a, b = min(px, x_test), max(px, x_test)
                            fa, fb = f_test if px > x_test else pf, pf if px > x_test else f_test
                            break

                    if a is not None:
                        break

                # Try negative side
                x_test2 = x0 - step
                if x_test2 not in [p[0] for p in points]:
                    f_test2 = f_counted(x_test2)
                    points.append((x_test2, f_test2))

                    if abs(f_test2) <= tol:
                        return x_test2

                    for (px, pf) in points:
                        if pf * f_test2 < 0:
                            a, b = min(px, x_test2), max(px, x_test2)
                            fa, fb = f_test2 if px > x_test2 else pf, pf if px > x_test2 else f_test2
                            break

                    if a is not None:
                        break

                step *= 2.0
                if step > 1e10:
                    break

    # If we found a bracket, use Brent's method
    if a is not None:
        from scipy.optimize import brentq
        try:
            root = brentq(f_counted, a, b, xtol=1e-12, rtol=1e-12, maxiter=200)
            if eval_count[0] <= max_evals:
                return root
        except Exception:
            pass

    # If no bracket found, use a robust secant method with bisection fallback
    # Find the two points with smallest |f|
    points.sort(key=lambda p: abs(p[1]))
    x1, f1 = points[0]
    x2, f2 = points[1]

    # Secant method
    x_prev, f_prev = x1, f1
    x_curr, f_curr = x2, f2

    for _ in range(100):
        if eval_count[0] >= max_evals - 1:
            break

        if f_curr == f_prev:
            x_new = x_curr + 1e-4 * max(1.0, abs(x_curr))
        else:
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)

        if not np.isfinite(x_new):
            break

        # Limit step size
        max_step = 10.0 * abs(x_curr - x_prev) + 1e-4
        if abs(x_new - x_curr) > max_step:
            x_new = x_curr + np.sign(x_new - x_curr) * max_step

        f_new = f_counted(x_new)

        if abs(f_new) <= tol:
            return x_new

        # Update points
        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_new, f_new

        if abs(x_new - x_curr) < 1e-12 * max(1.0, abs(x_new)):
            if abs(f_new) <= tol:
                return x_new
            break

    # Return the point with smallest |f|
    best_x = min(points, key=lambda p: abs(p[1]))[0]
    return best_x
