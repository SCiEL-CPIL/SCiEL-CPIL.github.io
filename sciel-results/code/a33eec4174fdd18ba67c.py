"""Root finding: Ill-conditioned - robust hybrid solver with Brent's method."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a portfolio: Brent's method (via bracketing), Newton with adaptive
    step, secant, and strategic restarts.
    """
    start_time = time.time()
    DEADLINE = 110.0
    MAX_EVALS = 500
    TOL = 1e-8

    eval_count = [0]
    best_x = float(x0)
    best_fx = None
    best_abs = float('inf')

    def check_time():
        return (time.time() - start_time) > DEADLINE

    def eval_f(x):
        nonlocal best_x, best_fx, best_abs
        if eval_count[0] >= MAX_EVALS:
            return best_fx if best_fx is not None else 0.0
        eval_count[0] += 1
        try:
            val = float(f(x))
        except Exception:
            val = 0.0
        if not math.isfinite(val):
            val = 0.0
        if abs(val) < best_abs:
            best_x = float(x)
            best_fx = val
            best_abs = abs(val)
        return val

    def is_good(x, fx=None):
        if fx is None:
            fx = eval_f(x)
        return abs(fx) <= TOL

    # --- Brent's method (requires bracket) ---
    def brent(a, b, fa, fb, max_iter=100):
        """Brent's method given a bracket [a,b] with fa*fb < 0."""
        nonlocal best_x, best_fx, best_abs
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        c = a
        fc = fa
        d = b - a
        e = d
        for _ in range(max_iter):
            if check_time() or eval_count[0] >= MAX_EVALS - 3:
                break
            if fa * fb > 0:
                break
            if fa * fc > 0:
                c = a
                fc = fa
                d = e = b - c
            if abs(fc) < abs(fb):
                a = b
                b = c
                c = a
                fa = fb
                fb = fc
                fc = fa
            tol1 = 2e-16 * max(1.0, abs(b))
            delta = 0.5 * (c - a)
            if abs(delta) <= tol1 or abs(fb) <= TOL:
                break
            if abs(e) >= tol1 and abs(fa) > abs(fb):
                s = fb / fa
                if a == c:
                    p = 2.0 * delta * s
                    q = 1.0 - s
                else:
                    q = fa / fc
                    r = fb / fc
                    p = s * (2.0 * delta * q * (q - r) - (b - a) * (r - 1.0))
                    q = (q - 1.0) * (r - 1.0) * (s - 1.0)
                if p > 0:
                    q = -q
                elif p < 0:
                    p = -p
                q = abs(q)
                if 2.0 * p < min(3.0 * delta * q - abs(tol1 * q), abs(e * q)):
                    e = d
                    d = p / q
                else:
                    d = delta
                    e = d
            else:
                d = delta
                e = d
            if d > 0:
                a = b
            else:
                a = b - d
            fa = fb
            fb = eval_f(b)
            if abs(fb) <= TOL:
                break
        # Return the point with smallest |f|
        if abs(fa) <= abs(fb):
            return a, fa
        else:
            return b, fb

    # --- Find a bracket around x0 ---
    def find_bracket(x0):
        """Try to find a bracket [a,b] with f(a)*f(b) < 0."""
        a = float(x0)
        fa = eval_f(a)
        if abs(fa) <= TOL:
            return a, a, fa, fa  # degenerate: already a root

        # Try expanding in both directions
        step = 1e-3 * max(1.0, abs(a))
        for direction in [1, -1]:
            b = a
            fb = fa
            for i in range(30):
                b = a + direction * step * (2.0 ** i)
                fb = eval_f(b)
                if abs(fb) <= TOL:
                    return b, b, fb, fb
                if fa * fb < 0:
                    return a, b, fa, fb
        return None

    # --- Newton with adaptive step and finite difference ---
    def newton(x0, max_iter=50):
        """Newton's method with damping and adaptive step."""
        x = float(x0)
        fx = eval_f(x)
        if abs(fx) <= TOL:
            return x, fx

        for i in range(max_iter):
            if check_time() or eval_count[0] >= MAX_EVALS - 5:
                break

            # Central finite difference
            h = 1e-7 * max(1.0, abs(x))
            fp = eval_f(x + h)
            fm = eval_f(x - h)
            d = (fp - fm) / (2.0 * h)

            if abs(d) < 1e-15:
                break

            dx = -fx / d
            # Limit step
            max_step = 10.0 * max(1.0, abs(x))
            if abs(dx) > max_step:
                dx = max_step * math.copysign(1.0, dx)

            x_new = x + dx
            fx_new = eval_f(x_new)

            if abs(fx_new) <= TOL:
                return x_new, fx_new

            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                # Try half step
                x_half = x + 0.5 * dx
                fx_half = eval_f(x_half)
                if abs(fx_half) <= TOL:
                    return x_half, fx_half
                if abs(fx_half) < abs(fx):
                    x, fx = x_half, fx_half
                else:
                    break
        return x, fx

    # --- Secant method ---
    def secant(x0, max_iter=40):
        """Secant method."""
        x0 = float(x0)
        x1 = x0 + 1e-4 * max(1.0, abs(x0))
        f0 = eval_f(x0)
        f1 = eval_f(x1)
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
            if not math.isfinite(x2):
                break
            f2 = eval_f(x2)
            if abs(f2) <= TOL:
                return x2, f2
            x0, f0 = x1, f1
            x1, f1 = x2, f2
        if abs(f0) <= abs(f1):
            return x0, f0
        else:
            return x1, f1

    # --- Bisection (simple, robust) ---
    def bisection(a, b, fa, fb, max_iter=80):
        """Simple bisection given a bracket."""
        for i in range(max_iter):
            if check_time() or eval_count[0] >= MAX_EVALS - 3:
                break
            c = 0.5 * (a + b)
            fc = eval_f(c)
            if abs(fc) <= TOL:
                return c, fc
            if fa * fc <= 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
            if abs(b - a) < 1e-14:
                break
        if abs(fa) <= abs(fb):
            return a, fa
        else:
            return b, fb

    # --- Execute strategies ---
    x0 = float(x0)
    f0 = eval_f(x0)
    if abs(f0) <= TOL:
        return x0

    # Strategy 1: Try to find a bracket and use Brent's method
    bracket = find_bracket(x0)
    if bracket is not None:
        a, b, fa, fb = bracket
        if a != b and fa * fb < 0:
            x, fx = brent(a, b, fa, fb)
            if abs(fx) <= TOL:
                return x
            # Also try bisection as fallback
            x, fx = bisection(a, b, fa, fb)
            if abs(fx) <= TOL:
                return x
        elif a == b:
            # Already found a root
            return a

    # Strategy 2: Newton from x0
    if not check_time() and eval_count[0] < MAX_EVALS - 10:
        x, fx = newton(x0)
        if abs(fx) <= TOL:
            return x

    # Strategy 3: Secant from x0
    if not check_time() and eval_count[0] < MAX_EVALS - 10:
        x, fx = secant(x0)
        if abs(fx) <= TOL:
            return x

    # Strategy 4: Newton from perturbed starts
    if not check_time() and eval_count[0] < MAX_EVALS - 10:
        for pert in [1e-3, -1e-3, 1e-2, -1e-2]:
            x_pert = x0 + pert * max(1.0, abs(x0))
            x, fx = newton(x_pert, max_iter=25)
            if abs(fx) <= TOL:
                return x
            if check_time() or eval_count[0] >= MAX_EVALS - 5:
                break

    # Strategy 5: Secant from perturbed starts
    if not check_time() and eval_count[0] < MAX_EVALS - 10:
        for pert in [1e-3, -1e-3]:
            x_pert = x0 + pert * max(1.0, abs(x0))
            x, fx = secant(x_pert, max_iter=25)
            if abs(fx) <= TOL:
                return x
            if check_time() or eval_count[0] >= MAX_EVALS - 5:
                break

    # Return best found
    return best_x

# EVOLVE-BLOCK-END
