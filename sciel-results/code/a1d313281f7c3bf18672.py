import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0
    max_calls = 480
    call_count = 0

    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    def f_call(x):
        nonlocal call_count
        call_count += 1
        return f(x)

    x = float(x0)
    if not math.isfinite(x):
        return x

    fx = f_call(x)

    if not math.isfinite(fx):
        return x

    if abs(fx) <= 1e-8:
        return x

    # Strategy: Find a bracket, then use secant-with-bisection.
    # If no bracket, use damped Newton.

    # Step 1: Find a bracket by expanding in both directions
    a = x
    fa = fx
    b = None
    fb = None

    # Try expanding right
    h = 1.0
    found_bracket = False
    for _ in range(15):
        if not time_ok():
            break
        b = a + h
        fb = f_call(b)
        if not math.isfinite(fb):
            h *= 2.0
            continue
        if fa * fb < 0:
            found_bracket = True
            break
        a = b
        fa = fb
        h *= 2.0

    if not found_bracket:
        # Try expanding left from original x0
        a = x
        fa = fx
        b = None
        fb = None
        h = 1.0
        for _ in range(15):
            if not time_ok():
                break
            b = a - h
            fb = f_call(b)
            if not math.isfinite(fb):
                h *= 2.0
                continue
            if fa * fb < 0:
                found_bracket = True
                break
            a = b
            fa = fb
            h *= 2.0

    if found_bracket and b is not None and fa * fb < 0:
        # We have a bracket [a, b] with f(a)*f(b) < 0
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Secant-with-bisection method
        # Maintain two points x0, x1 with f(x0) and f(x1) having opposite signs
        # Use secant step if it stays within bracket, else bisect
        
        x0 = a
        f0 = fa
        x1 = b
        f1 = fb

        for _ in range(200):
            if not time_ok():
                break

            if abs(f1) <= 1e-8:
                return x1
            if abs(f0) <= 1e-8:
                return x0

            # Check convergence
            if abs(x1 - x0) < 1e-12 * max(1.0, abs(x0), abs(x1)):
                return 0.5 * (x0 + x1)

            # Secant step
            if f1 != f0:
                x_secant = x1 - f1 * (x1 - x0) / (f1 - f0)
            else:
                x_secant = 0.5 * (x0 + x1)

            # Check if secant step is within bracket
            if x0 < x_secant < x1:
                x_next = x_secant
            else:
                x_next = 0.5 * (x0 + x1)

            f_next = f_call(x_next)

            if not math.isfinite(f_next):
                x_next = 0.5 * (x0 + x1)
                f_next = f_call(x_next)

            if abs(f_next) <= 1e-8:
                return x_next

            # Update bracket: keep the point with opposite sign to f_next
            if f0 * f_next < 0:
                x1 = x_next
                f1 = f_next
            else:
                x0 = x_next
                f0 = f_next

        # Return the best point
        return x1 if abs(f1) < abs(f0) else x0

    else:
        # No bracket found. Use damped Newton's method.
        # Use secant-like approach: estimate derivative from current and previous point.
        
        x_curr = float(x0)
        f_curr = fx

        # Try a few steps with damped Newton
        # Estimate derivative using finite difference
        for _ in range(60):
            if not time_ok():
                break
            if abs(f_curr) <= 1e-8:
                return x_curr

            # Estimate derivative
            h = 1e-7 * max(1.0, abs(x_curr))
            f_plus = f_call(x_curr + h)
            
            if not math.isfinite(f_plus):
                # Try smaller h
                h = 1e-8 * max(1.0, abs(x_curr))
                f_plus = f_call(x_curr + h)
                if not math.isfinite(f_plus):
                    break

            df = (f_plus - f_curr) / h

            if df == 0 or not math.isfinite(df):
                break

            x_new = x_curr - f_curr / df

            if not math.isfinite(x_new):
                break

            # Damping: limit step size
            max_step = 10.0 * max(1.0, abs(x_curr))
            if abs(x_new - x_curr) > max_step:
                x_new = x_curr + max_step * math.copysign(1.0, x_new - x_curr)

            x_curr = x_new
            f_curr = f_call(x_curr)

        return x_curr
