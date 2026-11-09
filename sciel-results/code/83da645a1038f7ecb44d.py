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

    # Step 1: Find a bracket by expanding in both directions
    a = x
    fa = fx
    b = None
    fb = None

    # Try expanding right
    h = 1.0
    for _ in range(30):
        if not time_ok():
            break
        b = a + h
        fb = f_call(b)
        if not math.isfinite(fb):
            h *= 2.0
            continue
        if fa * fb < 0:
            break
        a = b
        fa = fb
        h *= 2.0

    if b is None or fa * fb >= 0:
        # Try expanding left from original x0
        a = x
        fa = fx
        b = None
        fb = None
        h = 1.0
        for _ in range(30):
            if not time_ok():
                break
            b = a - h
            fb = f_call(b)
            if not math.isfinite(fb):
                h *= 2.0
                continue
            if fa * fb < 0:
                break
            a = b
            fa = fb
            h *= 2.0

    if b is not None and fa * fb < 0:
        # We have a bracket [a, b] with f(a)*f(b) < 0
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Use a robust hybrid: secant when inside bracket, bisection otherwise
        # This is simpler and often just as fast for smooth polynomials
        x_prev = a
        fx_prev = fa
        x_curr = b
        fx_curr = fb

        for _ in range(200):
            if not time_ok():
                break
            if abs(fx_curr) <= 1e-8:
                return x_curr
            if abs(fx_prev) <= 1e-8:
                return x_prev
            if abs(b - a) < 1e-13 * max(1.0, abs(a), abs(b)):
                return 0.5 * (a + b)

            # Secant step
            if fx_curr != fx_prev:
                x_secant = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
            else:
                x_secant = 0.5 * (a + b)

            # Use secant if it's inside the bracket, otherwise bisect
            if a < x_secant < b:
                x_next = x_secant
            else:
                x_next = 0.5 * (a + b)

            f_next = f_call(x_next)

            if not math.isfinite(f_next):
                x_next = 0.5 * (a + b)
                f_next = f_call(x_next)

            if abs(f_next) <= 1e-8:
                return x_next

            x_prev = x_curr
            fx_prev = fx_curr
            x_curr = x_next
            fx_curr = f_next

            # Update bracket
            if fa * f_next < 0:
                b = x_next
                fb = f_next
            else:
                a = x_next
                fa = f_next

        # Return the better of the last two points
        return x_curr if abs(fx_curr) < abs(fx_prev) else x_prev

    else:
        # No bracket found. Try Newton's method with finite differences.
        x = float(x0)
        fx = f_call(x)
        
        for _ in range(80):
            if not time_ok():
                break
            if abs(fx) <= 1e-8:
                return x
                
            h = 1e-7 * max(1.0, abs(x))
            fx_plus = f_call(x + h)
            d = (fx_plus - fx) / h
            
            if d == 0 or not math.isfinite(d):
                break
                
            x_new = x - fx / d
            
            if not math.isfinite(x_new):
                break
                
            # Damping
            if abs(x_new - x) > 10.0:
                x_new = x + 10.0 * math.copysign(1.0, x_new - x)
                
            x = x_new
            fx = f_call(x)
            
        return x
