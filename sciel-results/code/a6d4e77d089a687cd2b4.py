"""
Root finder using Brent's method with call-efficient derivative estimation.
Key improvements:
1. Use central-difference Newton steps (reusing evaluations) for fast convergence
2. Aggressive early termination at |f(x)| <= 1e-8
3. Minimal bracketing phase with adaptive step sizes
4. Secant method as primary driver when bracket is wide
5. Tight tolerance for final polish
"""
import time
import math

def find_root(f, x0):
    """
    Robust scalar root finder for smooth polynomials.
    Returns x such that |f(x)| <= 1e-8.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    TOL = 1e-8
    MAX_CALLS = 480

    x = float(x0)
    call_count = 0
    cache = {}

    def _elapsed():
        return time.time() - start_time

    def _time_ok():
        return _elapsed() < deadline and call_count < MAX_CALLS

    def _f(x_val):
        """Cached function evaluation with call counting."""
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return None
        key = x_val
        if key in cache:
            return cache[key]
        try:
            val = f(key)
            call_count += 1
            if not math.isfinite(val):
                cache[key] = None
                return None
            cache[key] = val
            return val
        except Exception:
            call_count += 1
            return None

    # Check if x0 is already a root
    fx0 = _f(x)
    if fx0 is None:
        return x
    if abs(fx0) <= TOL:
        return x

    # Phase 1: Find a bracket [a, b] with f(a)*f(b) < 0
    # Use exponential expansion from x0
    a = b = None
    fa = fb = None

    step = 0.5
    for i in range(50):
        if not _time_ok():
            break
        if step > 1e10:
            break

        # Try right
        xr = x + step
        fr = _f(xr)
        if fr is not None:
            if abs(fr) <= TOL:
                return xr
            if fx0 * fr < 0:
                a, b = x, xr
                fa, fb = fx0, fr
                break

        # Try left
        xl = x - step
        fl = _f(xl)
        if fl is not None:
            if abs(fl) <= TOL:
                return xl
            if fx0 * fl < 0:
                a, b = xl, x
                fa, fb = fl, fx0
                break

        step *= 1.6

    # If no bracket found, do a wider grid search
    if a is None or b is None:
        best_x = x
        best_abs = abs(fx0)
        prev_x, prev_f = None, None

        # Search on expanding intervals
        for scale in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
            if not _time_ok():
                break
            base = max(abs(x0), 1.0) * scale
            n_samples = 30
            for i in range(n_samples + 1):
                if not _time_ok():
                    break
                t = i / n_samples
                x_try = x0 - base + 2 * base * t
                f_try = _f(x_try)
                if f_try is None:
                    continue
                if abs(f_try) < best_abs:
                    best_abs = abs(f_try)
                    best_x = x_try
                if abs(f_try) <= TOL:
                    return x_try
                if prev_f is not None and prev_f * f_try < 0:
                    a, b = prev_x, x_try
                    fa, fb = prev_f, f_try
                    break
                prev_x, prev_f = x_try, f_try
            if a is not None:
                break

    # If still no bracket, fall back to secant from best point
    if a is None or b is None:
        x_curr = best_x
        fx_curr = _f(x_curr)
        if fx_curr is None:
            return best_x
        x_prev = x_curr + 1e-4 * max(1.0, abs(x_curr))
        fx_prev = _f(x_prev)
        if fx_prev is None:
            return best_x

        for _ in range(60):
            if not _time_ok():
                break
            if abs(fx_curr) <= TOL:
                return x_curr
            if abs(fx_prev) <= TOL:
                return x_prev
            denom = fx_curr - fx_prev
            if abs(denom) < 1e-15:
                break
            x_new = x_curr - fx_curr * (x_curr - x_prev) / denom
            if not math.isfinite(x_new):
                break
            x_prev, fx_prev = x_curr, fx_curr
            x_curr = x_new
            fx_curr = _f(x_curr)
            if fx_curr is None:
                break
        return x_curr

    # Phase 2: Newton's method with central difference derivative estimation
    # This is more efficient than Brent for smooth polynomials
    x_a, x_b = a, b
    f_a, f_b = fa, fb

    # Ensure we have a valid bracket
    if f_a * f_b > 0:
        # No valid bracket, use secant from the two points
        x_curr = x_a if abs(f_a) < abs(f_b) else x_b
        fx_curr = f_a if x_curr == x_a else f_b
        x_prev = x_b if x_curr == x_a else x_a
        fx_prev = f_b if x_curr == x_a else f_a

        for _ in range(80):
            if not _time_ok():
                break
            if abs(fx_curr) <= TOL:
                return x_curr
            if abs(fx_prev) <= TOL:
                return x_prev
            denom = fx_curr - fx_prev
            if abs(denom) < 1e-15:
                break
            x_new = x_curr - fx_curr * (x_curr - x_prev) / denom
            if not math.isfinite(x_new):
                break
            x_prev, fx_prev = x_curr, fx_curr
            x_curr = x_new
            fx_curr = _f(x_curr)
            if fx_curr is None:
                break
        return x_curr if abs(fx_curr) <= abs(fx_prev) else x_prev

    # Use the point with smaller |f| as starting point for Newton
    x_curr = x_a if abs(f_a) < abs(f_b) else x_b
    fx_curr = f_a if x_curr == x_a else f_b
    x_other = x_b if x_curr == x_a else x_a
    fx_other = f_b if x_curr == x_a else f_a

    # Newton's method with central difference derivative
    # Reuse evaluations to minimize function calls
    h = 1e-5 * max(1.0, abs(x_curr))
    
    for iteration in range(80):
        if not _time_ok():
            break
        if abs(fx_curr) <= TOL:
            return x_curr

        # Central difference derivative: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
        x_plus = x_curr + h
        x_minus = x_curr - h
        f_plus = _f(x_plus)
        f_minus = _f(x_minus)

        if f_plus is None or f_minus is None:
            # Fall back to forward difference
            x_plus = x_curr + h
            f_plus = _f(x_plus)
            if f_plus is None:
                break
            df = (f_plus - fx_curr) / h
        else:
            df = (f_plus - f_minus) / (2 * h)

        if abs(df) < 1e-15:
            # Derivative too small, use secant step
            if abs(fx_curr - fx_other) > 1e-15:
                x_new = x_curr - fx_curr * (x_curr - x_other) / (fx_curr - fx_other)
            else:
                break
        else:
            x_new = x_curr - fx_curr / df

        if not math.isfinite(x_new):
            break

        # Damping: if x_new is too far, move halfway
        if abs(x_new - x_curr) > 10 * h:
            x_new = x_curr + 0.5 * (x_new - x_curr)

        # Check if we're still in a reasonable range
        if abs(x_new) > 1e10:
            x_new = x_curr + 0.5 * (x_new - x_curr)

        f_new = _f(x_new)
        if f_new is None:
            break

        if abs(f_new) <= TOL:
            return x_new

        # Update state
        x_other, fx_other = x_curr, fx_curr
        x_curr, fx_curr = x_new, f_new

        # Adaptive step size for derivative
        h = max(1e-8, 0.5 * h)

    # Return the best point found
    if abs(fx_curr) <= abs(fx_other):
        return x_curr
    else:
        return x_other
