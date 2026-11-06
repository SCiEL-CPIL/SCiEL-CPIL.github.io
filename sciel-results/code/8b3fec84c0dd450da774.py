"""Root finding for ill-conditioned functions."""
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a hybrid strategy:
      1. Try to bracket the root via adaptive step-doubling.
      2. Use Illinois method (modified secant) for robust convergence.
      3. If no bracket found, fall back to secant/Newton-like iterations.
    Includes time guard to finish within deadline.
    """
    x0 = float(x0)
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    MAX_EVALS = 500
    tol = 1e-8
    best_x = x0
    best_fval = float('inf')

    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE

    def update_best(x, fx):
        nonlocal best_x, best_fval
        if abs(fx) < best_fval:
            best_fval = abs(fx)
            best_x = x
        if abs(fx) <= tol:
            return x
        return None

    eval_count = 0

    def feval(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS or not elapsed_ok():
            return None
        eval_count += 1
        try:
            val = f(x)
        except Exception:
            return None
        if val is None or (isinstance(val, float) and (val != val)):  # NaN check
            return None
        return float(val)

    # Check initial point
    fx0 = feval(x0)
    if fx0 is None:
        return best_x
    result = update_best(x0, fx0)
    if result is not None:
        return result

    # Try to find a bracket [a, b] with sign change
    a = x0
    fa = fx0
    # Expand to find sign change
    step = 1e-3 * max(1.0, abs(x0))
    b = x0 + step
    fb = feval(b)
    if fb is None:
        return best_x
    update_best(b, fb)

    bracketed = False
    for _ in range(50):
        if not elapsed_ok() or eval_count >= MAX_EVALS - 10:
            break
        if fa * fb < 0:
            bracketed = True
            break
        # Double the step
        step *= 2.0
        b = x0 + step
        fb = feval(b)
        if fb is None:
            break
        update_best(b, fb)
        if abs(fb) <= tol:
            return b
        # Also try expanding in the other direction
        b2 = x0 - step
        fb2 = feval(b2)
        if fb2 is None:
            break
        update_best(b2, fb2)
        if abs(fb2) <= tol:
            return b2
        if fa * fb2 < 0:
            a, b = x0, b2
            fa, fb = fx0, fb2
            bracketed = True
            break

    if bracketed:
        # Use Illinois method on [a, b]
        # Ensure |fa| <= |fb| so we keep the smaller-magnitude side
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        for _ in range(200):
            if not elapsed_ok() or eval_count >= MAX_EVALS:
                break
            # Secant interpolation
            if fb != fa:
                x_new = a - fa * (b - a) / (fb - fa)
            else:
                x_new = 0.5 * (a + b)
            fx_new = feval(x_new)
            if fx_new is None:
                break
            res = update_best(x_new, fx_new)
            if res is not None:
                return res
            # Update bracket
            if fa * fx_new < 0:
                b = x_new
                fb = fx_new
            else:
                # Replace the larger-magnitude endpoint
                a = x_new
                fa = fx_new
                # Illinois: shrink the side we just replaced
                if abs(fa) < abs(fb):
                    b = a + 0.5 * (b - a)
                    fb = feval(b)
                    if fb is None:
                        break
                    update_best(b, fb)
            if abs(b - a) < 1e-15 * max(1.0, abs(a)):
                break
        return best_x
    else:
        # No bracket found: use secant method iteratively
        x_prev = x0
        f_prev = fx0
        # Try a nearby point for secant
        x_curr = x0 + 1e-6 * max(1.0, abs(x0))
        f_curr = feval(x_curr)
        if f_curr is None:
            return best_x
        update_best(x_curr, f_curr)
        for _ in range(200):
            if not elapsed_ok() or eval_count >= MAX_EVALS:
                break
            if f_curr == f_prev:
                # Degenerate: try Newton with finite difference
                h = 1e-8 * max(1.0, abs(x_curr))
                f_plus = feval(x_curr + h)
                if f_plus is None:
                    break
                d = (f_plus - f_curr) / h
                if d == 0:
                    break
                x_new = x_curr - f_curr / d
            else:
                x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            f_new = feval(x_new)
            if f_new is None:
                break
            res = update_best(x_new, f_new)
            if res is not None:
                return res
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_new, f_new
        return best_x

# EVOLVE-BLOCK-END
