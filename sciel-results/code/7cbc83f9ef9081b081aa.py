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

    # Strategy:
    # 1. Try to find a bracket by expanding in both directions.
    # 2. Use Brent's method (properly implemented) to converge quickly.
    # 3. If no bracket found, try Newton with finite differences as fallback.

    # Step 1: Find a bracket
    a = x
    fa = fx
    b = None
    fb = None

    # Try expanding right
    h = 1.0
    for _ in range(20):
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
        for _ in range(20):
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

        # Proper Brent's method implementation
        # Based on the standard algorithm from Numerical Recipes / Brent's original paper
        c = a
        fc = fa
        mflag = True
        e = 0.0  # previous step

        for _ in range(200):
            if not time_ok():
                break

            if abs(fb) <= 1e-8:
                return b
            if abs(fa) <= 1e-8:
                return a

            # Check convergence
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                return 0.5 * (a + b)

            # Inverse quadratic interpolation if possible
            if mflag:
                d = (fb - fc) * (fb - fa)
                m = (b - c) * (fc - fa) - (b - a) * (fb - fc)
                mflag = False
            else:
                m = (b - c) * (fc - fa) - (b - a) * (fb - fc)

            if 2 * m < min(abs(d), abs(b - a) * (fb - fa)):
                # Interpolation is acceptable
                if abs(m) > 1e-300:
                    s = m / d
                else:
                    s = 0.5 * (b - a)
            else:
                s = 0.5 * (b - a)
                mflag = True

            # Check if the step is too small; use bisection
            if (s < 0.5 * (a - b) and s > 0.5 * (a - b) * -1) or \
               (s < 0.5 * (b - a) and s > 0.5 * (b - a) * -1):
                s = 0.5 * (b - a)
                mflag = True

            # Update c
            c = a
            fc = fa

            # Take the step
            if abs(s) >= abs(0.5 * (b - a)) - 1e-12:
                s = 0.5 * (b - a) if (b - a) > 0 else -0.5 * (b - a)

            # New point
            a = b
            fa = fb
            b = b + s
            fb = f_call(b)

            if not math.isfinite(fb):
                # Fallback: bisect
                b = 0.5 * (a + b)
                fb = f_call(b)

            if fa * fb < 0:
                # Root is in [a, b]
                continue
            else:
                # Root is not in [a, b], so update a
                a = c
                fa = fc
                c = b
                fc = fb
                # Swap to maintain a < b
                if a > b:
                    a, b = b, a
                    fa, fb = fb, fa

        # Return the best point
        return b if abs(fb) < abs(fa) else a

    else:
        # No bracket found. Try Newton's method with finite differences.
        x = float(x0)
        fx = f_call(x)
        
        for _ in range(60):
            if not time_ok():
                break
            if abs(fx) <= 1e-8:
                return x
                
            h = 1e-6 * max(1.0, abs(x))
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
