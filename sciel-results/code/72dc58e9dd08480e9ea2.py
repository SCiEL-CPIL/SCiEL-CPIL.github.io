"""Root finding: Ill-conditioned - Brent's method with aggressive bracketing and multi-start."""
import math
import time

# EVOLVE-BLOG-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Structural change: Brent's method (bisection + secant + ICI hybrid) as primary
    solver with geometric bracket expansion and multi-start.
    """
    start_time = time.time()
    MAX_WALL = 115.0
    MAX_EVALS = 500
    TOL = 1e-8
    X_EPS = 1e-14

    eval_count = [0]
    best_x = [float(x0)]
    best_f = [float('inf')]

    def eval_f(x):
        if eval_count[0] >= MAX_EVALS:
            raise RuntimeError("Max evaluations exceeded")
        if time.time() - start_time > MAX_WALL:
            raise RuntimeError("Time limit exceeded")
        eval_count[0] += 1
        try:
            val = f(x)
        except Exception:
            val = 0.0
        if not math.isfinite(val):
            val = 0.0
        if abs(val) < best_f[0]:
            best_f[0] = abs(val)
            best_x[0] = float(x)
        return val

    def ok():
        return eval_count[0] < MAX_EVALS and (time.time() - start_time < MAX_WALL)

    def is_good(x, fx):
        return abs(fx) <= TOL

    # --- Brent's method (full implementation) ---
    def brent(a, fa, b, fb, max_iter=80):
        """
        Brent's root-finding method.
        Combines bisection, secant, and inverse quadratic interpolation.
        Requires f(a)*f(b) < 0.
        Returns root or None.
        """
        if fa * fb > 0:
            return None
        if abs(fa) <= TOL:
            return a
        if abs(fb) <= TOL:
            return b

        # Ensure |f(a)| <= |f(b)|
        if abs(fa) < abs(fb):
            a, fa = b, fb
            b, fb = a, fa

        # Initial guess
        c = a
        fc = fa
        d = b - a
        e = d

        for _ in range(max_iter):
            if not ok():
                break

            if abs(fb) <= TOL:
                return b
            if abs(b - a) <= X_EPS * max(1.0, abs(b)):
                return b

            # Inverse quadratic interpolation
            try:
                if c != a and c != b and fc != fa and fc != fb:
                    # ICI
                    s = (b / (b - c)) * (fb / (fb - fc))
                    t = (b / (b - a)) * (fb / (fb - fa))
                    u = (c / (c - a)) * (fc / (fc - fa))
                    x_new = b + (s * t * u) / ((s + t) * (s + u))
                elif c != a and c != b and fc != fa and fc != fb:
                    x_new = b + (s * t * u) / ((s + t) * (s + u))
                else:
                    # Secant
                    if fa != fb:
                        x_new = b - fb * (b - a) / (fb - fa)
                    else:
                        x_new = 0.5 * (a + b)
            except (ZeroDivisionError, OverflowError):
                x_new = 0.5 * (a + b)

            if not math.isfinite(x_new):
                x_new = 0.5 * (a + b)

            # Conditions for using the new estimate
            cond1 = not ((min(a, b) < x_new < max(a, b)) or
                         (x_new < min(a, b)) != (x_new > max(a, b)))
            cond2 = abs(x_new - b) >= 0.5 * abs(b - c)
            cond3 = abs(x_new - b) >= X_EPS * max(1.0, abs(b))

            if cond1 or cond2 or cond3:
                # Fall back to bisection
                e = d
                d = 0.5 * (b - a)
                x_new = b + d
            else:
                d = x_new - b

            fx_new = eval_f(x_new)

            if is_good(x_new, fx_new):
                return x_new

            # Update
            if fa * fx_new < 0:
                b, fb = a, fa
            a, fa = b, fb
            b, fb = x_new, fx_new

            if abs(fa) < abs(fb):
                b, fb = a, fa
                a, fa = b, fb

            c, fc = b, fb
            if abs(fa) < abs(fb):
                c, fc = a, fa

            if abs(d) < X_EPS * max(1.0, abs(b)):
                break

        if abs(fb) <= TOL:
            return b
        if abs(fa) <= TOL:
            return a
        return None

    # --- Secant method (for polishing) ---
    def secant(xa, fa, xb, fb, max_iter=30):
        for _ in range(max_iter):
            if not ok():
                break
            if abs(fa) <= TOL:
                return xa
            if abs(fb) <= TOL:
                return xb
            if fa == fb:
                break
            if abs(xa - xb) < 1e-15 * max(1.0, abs(xa)):
                break
            d = (fa - fb) / (xa - xb)
            if d == 0 or not math.isfinite(d):
                break
            xc = xa - fa / d
            if not math.isfinite(xc):
                break
            fc = eval_f(xc)
            if is_good(xc, fc):
                return xc
            xa, fa = xb, fb
            xb, fb = xc, fc
        if abs(fa) <= TOL:
            return xa
        if abs(fb) <= TOL:
            return xb
        return None

    # --- Geometric bracket finding ---
    def find_bracket(x_start, max_steps=25):
        """
        Find a bracket [a,b] with f(a)*f(b) < 0 using geometrically growing steps.
        Returns (a, fa, b, fb) or None.
        """
        if not ok():
            return None

        fa = eval_f(x_start)
        if is_good(x_start, fa):
            return (x_start, fa, x_start, fa)

        base = max(1.0, abs(x_start))

        # Try geometrically growing steps in both directions
        # Steps: 1e-3, 1e-2, 1e-1, 1, 10, 100, ...
        for i in range(max_steps):
            if not ok():
                break
            step = base * (10 ** (i - 3))  # 1e-3, 1e-2, 1e-1, 1, 10, ...
            if step < 1e-15:
                step = 1e-15

            # Try right
            xr = x_start + step
            fr = eval_f(xr)
            if is_good(xr, fr):
                return (xr, fr, xr, fr)
            if fa * fr < 0:
                return (x_start, fa, xr, fr)

            # Try left
            xl = x_start - step
            fl = eval_f(xl)
            if is_good(xl, fl):
                return (xl, fl, xl, fl)
            if fa * fl < 0:
                return (xl, fl, x_start, fa)

        return None

    # --- Multi-start bracket finding ---
    def find_bracket_multi(x0, max_starts=5):
        """
        Try to find a bracket from multiple starting points.
        Returns (a, fa, b, fb) or None.
        """
        # First try from x0
        bracket = find_bracket(x0, max_steps=20)
        if bracket is not None:
            return bracket

        # Try from nearby points
        base = max(1.0, abs(x0))
        offsets = [
            1e-4 * base, -1e-4 * base,
            1e-3 * base, -1e-3 * base,
            1e-2 * base, -1e-2 * base,
            1e-1 * base, -1e-1 * base,
            1.0 * base, -1.0 * base,
        ]

        for off in offsets[:max_starts * 2]:
            if not ok():
                break
            x_try = x0 + off
            bracket = find_bracket(x_try, max_steps=15)
            if bracket is not None:
                return bracket

        return None

    # === MAIN EXECUTION ===

    # 1. Quick check at x0
    fx0 = eval_f(float(x0))
    if is_good(x0, fx0):
        return float(x0)

    # 2. Try secant from x0 with a nearby point (fast path)
    h = 1e-3 * max(1.0, abs(x0))
    if ok():
        x1 = x0 + h
        fx1 = eval_f(x1)
        if is_good(x1, fx1):
            return x1
        r = secant(x0, fx0, x1, fx1, max_iter=40)
        if r is not None:
            return r

    # 3. Find a bracket (geometric + multi-start)
    bracket = find_bracket_multi(x0, max_starts=4)

    if bracket is not None:
        a, fa, b, fb = bracket
        if is_good(a, fa):
            return a
        if is_good(b, fb):
            return b
        if fa * fb < 0:
            # Use Brent's method
            r = brent(a, fa, b, fb, max_iter=80)
            if r is not None:
                return r
        else:
            # Same sign but we have two points; try secant
            r = secant(a, fa, b, fb, max_iter=40)
            if r is not None:
                return r

    # 4. If no bracket found, try more aggressive multi-start
    if ok():
        bracket2 = find_bracket_multi(x0, max_starts=5)
        if bracket2 is not None:
            a, fa, b, fb = bracket2
            if is_good(a, fa):
                return a
            if is_good(b, fb):
                return b
            if fa * fb < 0:
                r = brent(a, fa, b, fb, max_iter=60)
                if r is not None:
                    return r

    # 5. Final secant polish from best point
    if ok():
        h2 = 1e-6 * max(1.0, abs(best_x[0]))
        fx_near = eval_f(best_x[0] + h2)
        if is_good(best_x[0] + h2, fx_near):
            return best_x[0] + h2
        r = secant(best_x[0], best_f[0], best_x[0] + h2, fx_near, max_iter=30)
        if r is not None:
            return r

    # Final fallback
    return best_x[0]

# EVOLVE-BLOG-END
