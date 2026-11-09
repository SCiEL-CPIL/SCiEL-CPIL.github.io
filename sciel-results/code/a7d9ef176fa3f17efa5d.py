"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    deadline = time.time() + 145.0
    tol = 1e-10  # target |f(x)| <= 1e-8, use tighter internal tol
    max_calls = 480  # stay well under 500

    x0 = float(x0)
    calls = [0]

    def fc(x):
        calls[0] += 1
        try:
            val = f(x)
        except Exception:
            return float('nan')
        if val is None or not math.isfinite(val):
            return float('nan')
        return float(val)

    best_x = x0
    best_abs = float('inf')
    best_f = None

    def update_best(x_val, f_val):
        nonlocal best_x, best_abs, best_f
        if x_val is None or f_val is None:
            return
        if not math.isfinite(x_val) or not math.isfinite(f_val):
            return
        abs_f = abs(f_val)
        if abs_f < best_abs:
            best_x = x_val
            best_abs = abs_f
            best_f = f_val
        if abs_f <= tol:
            return True
        return False

    # Initial evaluation
    f0 = fc(x0)
    update_best(x0, f0)
    if abs(f0) <= tol:
        return best_x

    # Try to find a bracket around x0
    def find_bracket(x_start, f_start, max_iter=200):
        """Find a bracket [a, b] where f(a) and f(b) have opposite signs."""
        if not math.isfinite(f_start):
            return None
        a = x_start
        fa = f_start
        # Try expanding in both directions
        step = max(1e-6, abs(x_start) * 1e-6)
        # First try small steps to find sign change
        for direction in [1, -1]:
            b = a + direction * step
            fb = fc(b)
            if not math.isfinite(fb):
                continue
            if update_best(b, fb):
                return (a, b, fa, fb)
            if fa * fb < 0:
                return (a, b, fa, fb)
            # Expand step exponentially
            for i in range(max_iter):
                if time.time() > deadline:
                    return None
                if calls[0] >= max_calls:
                    return None
                step *= 2.0
                b = a + direction * step
                fb = fc(b)
                if not math.isfinite(fb):
                    continue
                if update_best(b, fb):
                    return (a, b, fa, fb)
                if fa * fb < 0:
                    return (a, b, fa, fb)
                # Also check if we've gone too far
                if abs(b) > 1e15:
                    break
        return None

    # Bisection method
    def bisection(a, b, fa, fb, max_iter=100):
        nonlocal best_x, best_abs
        if fa * fb > 0:
            return False
        for i in range(max_iter):
            if time.time() > deadline:
                return False
            if calls[0] >= max_calls:
                return False
            mid = (a + b) / 2.0
            fm = fc(mid)
            if not math.isfinite(fm):
                return False
            if update_best(mid, fm):
                return True
            if abs(fm) <= tol:
                return True
            if fa * fm < 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
            if abs(b - a) < 1e-14 * max(1.0, abs(a)):
                break
        return False

    # Secant method
    def secant(xa, fa, xb, fb, max_iter=50):
        nonlocal best_x, best_abs
        for i in range(max_iter):
            if time.time() > deadline:
                return False
            if calls[0] >= max_calls:
                return False
            if abs(fb - fa) < 1e-15:
                return False
            xc = xb - fb * (xb - xa) / (fb - fa)
            if not math.isfinite(xc):
                return False
            fc_val = fc(xc)
            if not math.isfinite(fc_val):
                return False
            if update_best(xc, fc_val):
                return True
            if abs(fc_val) <= tol:
                return True
            # Keep the two points with smaller |f|
            if abs(fc_val) < abs(fa):
                xa, fa = xb, fb
                xb, fb = xc, fc_val
            else:
                xa, fa = xb, fb
                # keep xc as new point
                # Actually swap: we want the two most recent
                # Let's just use: old xa -> xb, xb -> xc
                # So: xa_new = xb, xb_new = xc
                # But we already did xa,fa = xb,fb above
                # Let me redo this logic
                pass
            # Simpler: always shift window
            xa, fa = xb, fb
            xb, fb = xc, fc_val
            if abs(xc - xb) < 1e-14 * max(1.0, abs(xc)):
                break
        return False

    # Newton with finite difference derivative
    def newton(x_start, f_start, max_iter=30):
        nonlocal best_x, best_abs
        x = x_start
        fx = f_start
        for i in range(max_iter):
            if time.time() > deadline:
                return False
            if calls[0] >= max_calls:
                return False
            if update_best(x, fx):
                return True
            h = 1e-7 * max(1.0, abs(x))
            fx_plus = fc(x + h)
            if not math.isfinite(fx_plus):
                return False
            d = (fx_plus - fx) / h
            if not math.isfinite(d) or abs(d) < 1e-15:
                return False
            x_new = x - fx / d
            if not math.isfinite(x_new):
                return False
            fx_new = fc(x_new)
            if not math.isfinite(fx_new):
                return False
            if update_best(x_new, fx_new):
                return True
            if abs(fx_new) <= tol:
                return True
            # Damping if step is too large
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x + 0.5 * (x_new - x)
                fx_new = fc(x_new)
                if not math.isfinite(fx_new):
                    return False
            x, fx = x_new, fx_new
            if abs(fx) < 1e-15:
                break
        return False

    # Strategy 1: Try to find a bracket and use bisection
    bracket = find_bracket(x0, f0)
    if bracket:
        a, b, fa, fb = bracket
        if bisection(a, b, fa, fb):
            return best_x

    # Strategy 2: Try secant from initial points
    step = max(1e-6, abs(x0) * 1e-6)
    x1 = x0 + step
    f1 = fc(x1)
    if math.isfinite(f1):
        update_best(x1, f1)
        if secant(x0, f0, x1, f1):
            return best_x

    # Strategy 3: Try Newton
    if newton(x0, f0):
        return best_x

    # Strategy 4: Multi-start with various offsets
    offsets = []
    base = max(1.0, abs(x0))
    for mult in [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
        offsets.append(mult * base)
    for offset in offsets:
        if time.time() > deadline:
            break
        if calls[0] >= max_calls:
            break
        for direction in [1, -1]:
            if time.time() > deadline:
                break
            if calls[0] >= max_calls:
                break
            x_try = x0 + direction * offset
            f_try = fc(x_try)
            if not math.isfinite(f_try):
                continue
            if update_best(x_try, f_try):
                return best_x
            # Try to bracket from here
            bracket = find_bracket(x_try, f_try, max_iter=50)
            if bracket:
                a, b, fa, fb = bracket
                if bisection(a, b, fa, fb):
                    return best_x
            # Try secant from x0 and x_try
            if math.isfinite(f0):
                if secant(x0, f0, x_try, f_try, max_iter=30):
                    return best_x
            # Try Newton from x_try
            if newton(x_try, f_try, max_iter=20):
                return best_x

    # Strategy 5: Dense local search for sign change
    if calls[0] < max_calls:
        n_dense = 50
        span = max(1.0, abs(x0)) * 0.1
        prev_x = x0 - span / 2
        prev_f = fc(prev_x)
        for i in range(1, n_dense + 1):
            if time.time() > deadline:
                break
            if calls[0] >= max_calls:
                break
            x_i = x0 - span / 2 + i * (span / n_dense)
            f_i = fc(x_i)
            if math.isfinite(prev_f) and math.isfinite(f_i):
                update_best(x_i, f_i)
                if prev_f * f_i < 0:
                    if bisection(prev_x, x_i, prev_f, f_i):
                        return best_x
            prev_x = x_i
            prev_f = f_i

    # Return best found
    if best_abs <= 1e-8:
        return best_x
    return best_x
# EVOLVE-BLOCK-END
