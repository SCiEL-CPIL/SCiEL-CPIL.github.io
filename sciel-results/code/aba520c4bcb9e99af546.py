"""Root finder for 3 fixed smooth polynomial equations."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def time_ok():
        return time.time() < deadline

    x = float(x0)
    best_x = x
    best_abs = float('inf')

    def update(xc, fc):
        nonlocal best_x, best_abs
        try:
            af = abs(fc)
        except Exception:
            af = float('inf')
        if math.isfinite(af) and af < best_abs:
            best_abs = af
            best_x = xc

    # --- Phase 1: Newton with finite-difference derivative ---
    for _ in range(40):
        if not time_ok():
            break
        try:
            fx = f(x)
        except Exception:
            break
        if not math.isfinite(fx):
            break
        update(x, fx)
        if abs(fx) <= 1e-12:
            return x
        h = 1e-7 * max(1.0, abs(x))
        try:
            d = (f(x + h) - fx) / h
        except Exception:
            break
        if not math.isfinite(d) or d == 0:
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        x = x_new

    if best_abs <= 1e-8:
        return best_x

    # --- Phase 2: find a bracket via geometric expansion ---
    a = b = None
    fa = fb = None
    try:
        fa = f(x0)
    except Exception:
        fa = None
    if fa is not None and math.isfinite(fa) and abs(fa) <= 1e-8:
        return x0

    scale = max(1.0, abs(x0))
    for k in range(1, 80):
        if not time_ok():
            break
        for sgn in (1, -1):
            try:
                cand = x0 + sgn * (2.0 ** k) * scale
                fc = f(cand)
            except Exception:
                fc = None
            if fc is not None and math.isfinite(fc):
                if abs(fc) <= 1e-8:
                    return cand
                if fa is not None and math.isfinite(fa) and fa * fc < 0:
                    a, b = x0, cand
                    fa, fb = fa, fc
                    break
        if a is not None:
            break

    # --- Phase 3: Illinois method to converge the bracket ---
    if a is not None:
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        for _ in range(200):
            if not time_ok():
                break
            if not math.isfinite(a) or not math.isfinite(b):
                break
            if b - a <= 1e-12 * max(1.0, abs(a), abs(b)):
                break
            # Illinois step: weighted midpoint favoring the smaller |f| side
            if abs(fa) < abs(fb):
                m = (a + b) / 2.0
            else:
                m = (a * abs(fb) + b * abs(fa)) / (abs(fa) + abs(fb))
            if not math.isfinite(m) or m == a or m == b:
                m = (a + b) / 2.0
            try:
                fm = f(m)
            except Exception:
                break
            if not math.isfinite(fm):
                break
            update(m, fm)
            if abs(fm) <= 1e-12:
                return m
            if fa * fm < 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
            if abs(fa) > abs(fb):
                a, b = b, a
                fa, fb = fb, fa

    if best_abs <= 1e-8:
        return best_x

    # --- Phase 4: final high-precision Newton polish from best point ---
    x = best_x
    for _ in range(30):
        if not time_ok():
            break
        try:
            fx = f(x)
        except Exception:
            break
        if not math.isfinite(fx):
            break
        update(x, fx)
        if abs(fx) <= 1e-14:
            return x
        h = 1e-8 * max(1.0, abs(x))
        try:
            d = (f(x + h) - fx) / h
        except Exception:
            break
        if not math.isfinite(d) or d == 0:
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        x = x_new

    return best_x
# EVOLVE-BLOCK-END
