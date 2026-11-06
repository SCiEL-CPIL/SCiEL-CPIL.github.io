"""Root finding: Ill-conditioned - robust multi-strategy solver."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a portfolio approach: Newton with damping, secant, bisection,
    and strategic restarts to handle ill-conditioned problems.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s wall-clock limit
    MAX_EVALS = 500
    TOL = 1e-8
    STRICT_TOL = 1e-12  # Tighter tolerance for early exit

    eval_count = [0]

    def eval_f(x):
        """Counted evaluation of f with finite-value guard."""
        eval_count[0] += 1
        try:
            val = f(x)
        except Exception:
            return 0.0
        if not math.isfinite(val):
            return 0.0
        return float(val)

    def is_good(x, fx=None):
        """Check if we've found a sufficiently good root."""
        if fx is None:
            fx = eval_f(x)
        return abs(fx) <= TOL

    def check_time():
        """Return True if we should stop due to time."""
        return (time.time() - start_time) > DEADLINE

    # Track best result found so far
    best_x = float(x0)
    best_fx = eval_f(best_x)
    best_abs = abs(best_fx)

    def update_best(x, fx):
        nonlocal best_x, best_fx, best_abs
        if abs(fx) < best_abs:
            best_x = x
            best_fx = fx
            best_abs = abs(fx)

    # --- Strategy 1: Damped Newton's method ---
    def try_newton(x0, max_iter=40):
        """Damped Newton with line search."""
        nonlocal best_x, best_fx, best_abs
        x = float(x0)
        fx = eval_f(x)
        update_best(x, fx)
        if abs(fx) <= TOL:
            return x, fx

        for i in range(max_iter):
            if check_time() or eval_count[0] >= MAX_EVALS - 5:
                break

            # Finite difference derivative (central if possible)
            h = 1e-7 * max(1.0, abs(x))
            fp = eval_f(x + h)
            fm = eval_f(x - h)
            d = (fp - fm) / (2.0 * h)

            if abs(d) < 1e-15:
                break

            # Newton step
            dx = -fx / d
            # Damping: limit step size
            max_step = 10.0 * max(1.0, abs(x))
            if abs(dx) > max_step:
                dx = max_step * math.copysign(1.0, dx)

            x_new = x + dx
            fx_new = eval_f(x_new)
            update_best(x_new, fx_new)

            if abs(fx_new) <= TOL:
                return x_new, fx_new

            # Accept if it reduces |f|, otherwise try smaller step
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                # Try half step
                x_half = x + 0.5 * dx
                fx_half = eval_f(x_half)
                update_best(x_half, fx_half)
                if abs(fx_half) <= TOL:
                    return x_half, fx_half
                if abs(fx_half) < abs(fx):
                    x, fx = x_half, fx_half
                else:
                    # Reject step, break
                    break

        return x, fx

    # --- Strategy 2: Secant method ---
    def try_secant(x0, max_iter=30):
        """Secant method with safeguards."""
        nonlocal best_x, best_fx, best_abs
        x0 = float(x0)
        x1 = x0 + 1e-4 * max(1.0, abs(x0))

        f0 = eval_f(x0)
        f1 = eval_f(x1)
        update_best(x0, f0)
        update_best(x1, f1)

        if abs(f0) <= TOL:
            return x0, f0
        if abs(f1) <= TOL:
            return x1, f1

        for i in range(max_iter):
            if check_time() or eval_count[0] >= MAX_EVALS - 5:
                break

            if abs(f1 - f0) < 1e-15:
                break

            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

            # Safeguard: keep in reasonable range
            if not math.isfinite(x2):
                break
            max_range = 100.0 * max(1.0, abs(x0))
            if abs(x2 - x0) > max_range:
                x2 = x0 + max_range * math.copysign(1.0, x2 - x0)

            f2 = eval_f(x2)
            update_best(x2, f2)

            if abs(f2) <= TOL:
                return x2, f2

            # Keep the two best points
            if abs(f2) < abs(f0):
                x0, f0 = x1, f1
            else:
                x0, f0 = x0, f0  # keep x0
            x1, f1 = x2, f2

            if abs(f1) < abs(f0):
                # swap so f1 is the smaller
                pass
            else:
                x0, f0, x1, f1 = x1, f1, x0, f0

        # Return best of the two
        if abs(f0) <= abs(f1):
            return x0, f0
        else:
            return x1, f1

    # --- Strategy 3: Bisection with bracket search ---
    def try_bisection(x0, max_iter=60):
        """Find a bracket then bisect."""
        nonlocal best_x, best_fx, best_abs
        a = float(x0)
        fa = eval_f(a)
        update_best(a, fa)
        if abs(fa) <= TOL:
            return a, fa

        # Try to find a bracket
        b = None
        fb = None
        step = 1e-4 * max(1.0, abs(a))
        for direction in [1, -1]:
            for i in range(20):
                b = a + direction * step * (2.0 ** i)
                fb = eval_f(b)
                update_best(b, fb)
                if abs(fb) <= TOL:
                    return b, fb
                if fa * fb < 0:
                    break
            if fa * fb < 0:
                break

        if b is None or fa * fb >= 0:
            return a, fa  # No bracket found

        # Bisection
        for i in range(max_iter):
            if check_time() or eval_count[0] >= MAX_EVALS - 5:
                break
            c = 0.5 * (a + b)
            fc = eval_f(c)
            update_best(c, fc)
            if abs(fc) <= TOL:
                return c, fc
            if fa * fc <= 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
            if abs(b - a) < 1e-14:
                break

        # Return midpoint or best
        mid = 0.5 * (a + b)
        fm = eval_f(mid)
        update_best(mid, fm)
        if abs(fm) <= abs(fa) and abs(fm) <= abs(fb):
            return mid, fm
        elif abs(fa) <= abs(fb):
            return a, fa
        else:
            return b, fb

    # --- Strategy 4: Steepest descent on f^2 ---
    def try_descent(x0, max_iter=30):
        """Gradient descent on g(x) = f(x)^2."""
        nonlocal best_x, best_fx, best_abs
        x = float(x0)
        fx = eval_f(x)
        update_best(x, fx)
        if abs(fx) <= TOL:
            return x, fx

        lr = 1e-3
        for i in range(max_iter):
            if check_time() or eval_count[0] >= MAX_EVALS - 5:
                break

            h = 1e-7 * max(1.0, abs(x))
            fp = eval_f(x + h)
            fm = eval_f(x - h)
            d = (fp - fm) / (2.0 * h)
            # Gradient of f^2 is 2*f*f'
            grad = 2.0 * fx * d

            if abs(grad) < 1e-15:
                break

            # Adaptive learning rate
            x_new = x - lr * grad
            fx_new = eval_f(x_new)
            update_best(x_new, fx_new)

            if abs(fx_new) <= TOL:
                return x_new, fx_new

            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
                lr = min(lr * 1.5, 1.0)
            else:
                lr *= 0.5
                if lr < 1e-12:
                    break

        return x, fx

    # Execute strategies in order, keeping the best result
    # Start with Newton (usually fastest for well-behaved functions)
    x, fx = try_newton(x0)
    if abs(fx) <= TOL:
        return x

    # Try secant from original x0
    x, fx = try_secant(x0)
    if abs(fx) <= TOL:
        return x

    # Try bisection (good for finding sign changes)
    x, fx = try_bisection(x0)
    if abs(fx) <= TOL:
        return x

    # Try descent
    x, fx = try_descent(x0)
    if abs(fx) <= TOL:
        return x

    # If none worked well, try Newton again from a perturbed start
    if eval_count[0] < MAX_EVALS - 10 and not check_time():
        x_pert = x0 + 1e-3 * max(1.0, abs(x0))
        x, fx = try_newton(x_pert, max_iter=20)
        if abs(fx) <= TOL:
            return x

        x_pert2 = x0 - 1e-3 * max(1.0, abs(x0))
        x, fx = try_newton(x_pert2, max_iter=20)
        if abs(fx) <= TOL:
            return x

    # Return the best result found
    return best_x

# EVOLVE-BLOCK-END
