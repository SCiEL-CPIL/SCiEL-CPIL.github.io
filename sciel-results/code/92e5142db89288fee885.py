import math
import time

def find_root(f, x0):
    """
    Find a root of f near x0 using Brent's method.
    Returns x with |f(x)| <= 1e-8.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    call_count = 0
    max_calls = 490  # Leave some margin

    def safe_f(x):
        nonlocal call_count
        call_count += 1
        try:
            val = f(float(x))
            if not math.isfinite(val):
                return float('nan')
            return float(val)
        except Exception:
            return float('nan')

    x0 = float(x0)
    fx0 = safe_f(x0)
    if abs(fx0) <= 1e-8:
        return x0

    # 1. Find a bracket
    a = x0
    fa = fx0
    b = None
    fb = None
    h = 1e-4
    max_h = 1e6

    for direction in [1, -1]:
        h = 1e-4
        while h <= max_h and call_count < max_calls:
            if time.time() > deadline:
                break
            x_try = x0 + direction * h
            fx_try = safe_f(x_try)
            if abs(fx_try) <= 1e-8:
                return x_try
            if fa * fx_try < 0:
                if direction == 1:
                    a, b = x0, x_try
                    fa, fb = fx0, fx_try
                else:
                    a, b = x_try, x0
                    fa, fb = fx_try, fx0
                break
            h *= 2.0
        if b is not None:
            break

    if b is None:
        # Fallback: Secant
        x1 = x0 + 1e-6
        fx1 = safe_f(x1)
        for _ in range(50):
            if time.time() > deadline or call_count >= max_calls:
                break
            if abs(fx1) <= 1e-8:
                return x1
            if fx1 == fa:
                break
            x_new = x1 - fx1 * (x1 - x0) / (fx1 - fa)
            x0, fa = x1, fx1
            x1 = x_new
            fx1 = safe_f(x1)
        return x0 if abs(fa) < abs(fx1) else x1

    # 2. Brent's Method
    # Ensure |fa| <= |fb|
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = b
    fc = fb
    d = b
    fd = fb

    tol_f = 1e-8
    eps = 2.22e-16

    for _ in range(100):
        if time.time() > deadline or call_count >= max_calls:
            break

        if abs(fb) <= tol_f:
            return b

        # Check if a and c have the same sign
        if fa * fc > 0:
            c = a
            fc = fa
            d = a
            fd = fa

        # If |fc| < |fa|, swap a and c
        if abs(fc) < abs(fa):
            a, c = c, a
            fa, fc = fc, fa

        # Tolerance
        tol = 2.0 * eps * max(abs(a), abs(b)) + 0.5 * tol_f

        if abs(b - a) <= tol:
            return b

        # Choose s
        s = None
        # Inverse Quadratic Interpolation
        if a != c and a != d:
            try:
                s = b + (b - a) * (fc / (fc - fa)) * (fd / (fd - fa))
            except ZeroDivisionError:
                s = None

            if s is not None:
                if not ( (b - max(a, b) / 2) < s < (min(a, b) + (b - a) / 2) ):
                    s = None

        # Secant
        if s is None and b != c:
            try:
                s = b - fb * (b - c) / (fb - fc)
            except ZeroDivisionError:
                s = None
            if s is not None:
                if not ( (b - max(a, b) / 2) < s < (min(a, b) + (b - a) / 2) ):
                    s = None

        # Bisection
        if s is None:
            s = (a + b) / 2.0
            d = (a + b) / 2.0
        else:
            d = b

        fs = safe_f(s)

        # Update bracket
        if fa * fs < 0:
            b = s
            fb = fs
        elif fc * fs < 0:
            a = c
            c = s
            fa = fc
            fc = fs
        elif fd * fs < 0:
            a = d
            d = s
            fa = fd
            fd = fs
        elif abs(fs) < abs(fb):
            a = b
            b = s
            fa = fb
            fb = fs
        else:
            c = s
            fc = fs
            d = b
            fd = fb

        if abs(fb) <= tol_f:
            return b

    # Return best
    if abs(fb) < abs(fa):
        return b
    else:
        return a
