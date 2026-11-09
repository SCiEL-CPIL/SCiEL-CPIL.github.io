import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 108.0
    max_calls = 420
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

    best_x = x
    best_f = fx

    # Phase 1: Try to find a bracket using geometric expansion
    a = x
    fa = fx
    b = None
    fb = None
    bracket_found = False

    for direction in [1, -1]:
        if not time_ok():
            break
        a = x
        fa = fx
        h = 0.5
        for _ in range(20):
            if not time_ok():
                break
            b = a + direction * h
            fb = f_call(b)
            if not math.isfinite(fb):
                h *= 2.0
                continue
            if abs(fb) < abs(best_f):
                best_x = b
                best_f = fb
            if fa * fb < 0:
                bracket_found = True
                break
            a = b
            fa = fb
            h *= 2.0
        if bracket_found:
            break

    if bracket_found:
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Brent's method: combines bisection, secant, and inverse quadratic interpolation
        # with guaranteed convergence via bisection fallback
        c = a
        fc = fa
        d = a
        dc = a
        e = b - a
        fe = fb

        for _ in range(100):
            if not time_ok():
                break

            if abs(fb) <= 1e-8:
                return b
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) < abs(best_f):
                best_x = b
                best_f = fb
            if abs(fa) < abs(best_f):
                best_x = a
                best_f = fa

            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                if abs(fa) < abs(fb):
                    return a
                else:
                    return b

            # Inverse quadratic interpolation if possible
            if fa != fc and fb != fc:
                s = a * fc * fb / ((fa - fc) * (fa - fb)) + \
                    b * fa * fc / ((fb - fc) * (fb - fa)) + \
                    c * fa * fb / ((fc - fa) * (fc - fb))
            elif e != 0:
                # Secant
                s = b - fb * (b - a) / (fb - fa)
            else:
                # Bisection
                s = 0.5 * (a + b)

            # Check conditions for using s
            m1 = (0.5 * (a + b) - s - e) * (b - a)
            m2 = (e - s) * (b - a)

            if (not (s > min(a, b) + 0.25 * (b - a) and s < max(a, b) - 0.25 * (b - a)) or
                    (m1 >= m2 * m2 and m1 >= e * e) or
                    (abs(s - b) >= abs(b - c) * 0.5)):
                # Use bisection
                s = 0.5 * (a + b)
                e = 0.5 * (b - a)
            else:
                # Use s
                if abs(s - b) >= abs(b - c) * 0.5:
                    e = 0.5 * (b - a)
                else:
                    e = s - b

            # Evaluate at s
            fs = f_call(s)

            if not math.isfinite(fs):
                s = 0.5 * (a + b)
                e = 0.5 * (b - a)
                fs = f_call(s)

            if abs(fs) <= 1e-8:
                return s
            if abs(fs) < abs(best_f):
                best_x = s
                best_f = fs

            # Update bracket
            if fa * fs < 0:
                c = b
                fc = fb
                b = s
                fb = fs
            else:
                c = a
                fc = fa
                a = s
                fa = fs

        return best_x
    else:
        # No bracket found. Use secant method (1 call per iteration after initial 2)
        x0_curr = x
        f0_curr = fx
        h = 1e-6 * max(1.0, abs(x))
        x1_curr = x + h
        f1_curr = f_call(x1_curr)

        if not math.isfinite(f1_curr):
            x1_curr = x - h
            f1_curr = f_call(x1_curr)

        if not math.isfinite(f1_curr):
            return best_x

        if abs(f1_curr) < abs(best_f):
            best_x = x1_curr
            best_f = f1_curr

        for _ in range(50):
            if not time_ok():
                break

            if abs(f1_curr) <= 1e-8:
                return x1_curr
            if abs(f0_curr) <= 1e-8:
                return x0_curr

            if f1_curr == f0_curr:
                # Degenerate: use bisection-like step
                x_new = 0.5 * (x0_curr + x1_curr)
            else:
                x_new = x1_curr - f1_curr * (x1_curr - x0_curr) / (f1_curr - f0_curr)

            if not math.isfinite(x_new):
                break

            # Damping
            step = x_new - x1_curr
            max_step = 5.0 * max(1.0, abs(x1_curr))
            if abs(step) > max_step:
                x_new = x1_curr + max_step * math.copysign(1.0, step)

            f_new = f_call(x_new)

            if not math.isfinite(f_new):
                break

            if abs(f_new) < abs(best_f):
                best_x = x_new
                best_f = f_new

            if abs(f_new) < abs(f1_curr):
                x0_curr = x1_curr
                f0_curr = f1_curr
                x1_curr = x_new
                f1_curr = f_new
            else:
                x0_curr = x_new
                f0_curr = f_new

        return best_x
