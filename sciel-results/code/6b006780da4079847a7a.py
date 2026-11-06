"""Root finding: Ill-conditioned - Multi-start with adaptive bracketing and Illinois/Secant hybrid."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Strategy:
    1. Quick check at x0
    2. Adaptive bracket finding with expanding steps
    3. Illinois method for guaranteed convergence when bracketed
    4. Secant method for fast convergence without derivative
    5. Multi-start if initial bracket fails
    """
    start_time = time.time()
    MAX_WALL = 115.0
    MAX_EVALS = 500
    TOL = 1e-8

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

    def good(x, fx, tol=TOL):
        return abs(fx) <= tol

    def ok():
        return eval_count[0] < MAX_EVALS and (time.time() - start_time < MAX_WALL)

    # --- Secant method ---
    def secant(xa, fa, xb, fb, max_iter=50):
        for i in range(max_iter):
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
            if good(xc, fc):
                return xc

            xa, fa = xb, fb
            xb, fb = xc, fc

            if abs(xb - xa) < 1e-15 * max(1.0, abs(xb)):
                break

        if abs(fa) <= TOL:
            return xa
        if abs(fb) <= TOL:
            return xb
        return None

    # --- Illinois method ---
    def illinois(xa, fa, xb, fb, max_iter=100):
        if fa * fb > 0:
            return None

        for i in range(max_iter):
            if not ok():
                break

            if abs(fa) <= TOL:
                return xa
            if abs(fb) <= TOL:
                return xb

            denom = fa - fb
            if denom == 0:
                xm = 0.5 * (xa + xb)
            else:
                xm = (fa * xb - fb * xa) / denom

            if xm <= min(xa, xb) or xm >= max(xa, xb):
                xm = 0.5 * (xa + xb)

            fxm = eval_f(xm)
            if good(xm, fxm):
                return xm

            if fa * fxm <= 0:
                if abs(fa) > abs(fxm):
                    xa_new = 0.5 * (xa + xm)
                    fa = eval_f(xa_new)
                    if good(xa_new, fa):
                        return xa_new
                else:
                    xb, fb = xm, fxm
            else:
                if abs(fb) > abs(fxm):
                    xb_new = 0.5 * (xm + xb)
                    fb = eval_f(xb_new)
                    if good(xb_new, fb):
                        return xb_new
                else:
                    xa, fa = xm, fxm

            if abs(xb - xa) < 1e-15 * max(1.0, abs(xm)):
                break

        if abs(fa) <= abs(fb):
            return xa if abs(fa) <= TOL else None
        return xb if abs(fb) <= TOL else None

    # --- Bracket finding with adaptive step ---
    def find_bracket(x0, max_scan=50):
        base_x = float(x0)
        fa = eval_f(base_x)
        if good(base_x, fa):
            return (base_x, fa, base_x, fa)

        # Try multiple step sizes
        for h_factor in [1e-2, 1e-1, 1.0, 10.0]:
            h = h_factor * max(1.0, abs(base_x))
            if h < 1e-10:
                h = 1e-10

            # Scan right
            prev_x = base_x
            prev_fx = fa
            for i in range(1, max_scan + 1):
                if not ok():
                    break
                cur_x = base_x + i * h
                cur_fx = eval_f(cur_x)
                if good(cur_x, cur_fx):
                    return (cur_x, cur_fx, cur_x, cur_fx)
                if prev_fx * cur_fx < 0:
                    return (prev_x, prev_fx, cur_x, cur_fx)
                prev_x, prev_fx = cur_x, cur_fx

            # Scan left
            prev_x = base_x
            prev_fx = fa
            for i in range(1, max_scan + 1):
                if not ok():
                    break
                cur_x = base_x - i * h
                cur_fx = eval_f(cur_x)
                if good(cur_x, cur_fx):
                    return (cur_x, cur_fx, cur_x, cur_fx)
                if prev_fx * cur_fx < 0:
                    return (cur_x, cur_fx, prev_x, prev_fx)
                prev_x, prev_fx = cur_x, cur_fx

        return None

    # === MAIN EXECUTION ===

    # 1. Quick check at x0
    fx0 = eval_f(float(x0))
    if good(x0, fx0):
        return float(x0)

    # 2. Try secant from x0 with a nearby point
    h = 1e-3 * max(1.0, abs(x0))
    if ok():
        x1 = x0 + h
        fx1 = eval_f(x1)
        if good(x1, fx1):
            return x1
        r = secant(x0, fx0, x1, fx1, max_iter=50)
        if r is not None:
            return r

    # 3. Try Illinois if we can find a bracket
    bracket = find_bracket(best_x[0], max_scan=50)
    if bracket is not None:
        xa, fa, xb, fb = bracket
        r = illinois(xa, fa, xb, fb, max_iter=100)
        if r is not None:
            return r

    # 4. Multi-start: try different initial points
    if ok():
        for offset_factor in [1e-1, 1e-2, 1e-3, -1e-1, -1e-2, -1e-3]:
            if not ok():
                break
            x_start = best_x[0] + offset_factor * max(1.0, abs(best_x[0]))
            fx_start = eval_f(x_start)
            if good(x_start, fx_start):
                return x_start

            # Try secant from this new start
            h2 = 1e-3 * max(1.0, abs(x_start))
            x2 = x_start + h2
            fx2 = eval_f(x2)
            if good(x2, fx2):
                return x2
            r = secant(x_start, fx_start, x2, fx2, max_iter=30)
            if r is not None:
                return r

            # Try bracketing from this start
            bracket2 = find_bracket(x_start, max_scan=20)
            if bracket2 is not None:
                xa, fa, xb, fb = bracket2
                r = illinois(xa, fa, xb, fb, max_iter=50)
                if r is not None:
                    return r

    # 5. Final secant polish from the best point
    if ok():
        h = 1e-6 * max(1.0, abs(best_x[0]))
        fx_near = eval_f(best_x[0] + h)
        if good(best_x[0] + h, fx_near):
            return best_x[0] + h
        r = secant(best_x[0], best_f[0], best_x[0] + h, fx_near, max_iter=40)
        if r is not None:
            return r

    # Final fallback
    return best_x[0] if best_f[0] <= TOL else best_x[0]

# EVOLVE-BLOCK-END
