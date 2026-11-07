import math
import time
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    x = float(x0)
    fx = f(x)

    # Check if x0 is already a root
    if abs(fx) <= 1e-8:
        return x

    best_x = x
    best_f = fx
    best_abs = abs(fx)

    fa = fx

    bracket_found = False
    lo = 0.0
    hi = 0.0
    flo = 0.0
    fhi = 0.0

    # Generate a denser set of deltas to catch narrow oscillations.
    # Use a geometric progression with ratio ~2.5 to balance coverage and call count.
    # Range: 1e-8 to 1e6
    deltas = []
    d = 1e-8
    while d <= 1e6:
        deltas.append(d)
        d *= 2.5

    # Also add the original power-of-10 steps for robustness at wide scales
    for i in range(-8, 7):
        v = 10.0 ** i
        if v not in deltas:
            deltas.append(v)
    deltas.sort()

    for d in deltas:
        if time.time() > deadline:
            break

        # Check right side
        x_right = x + d
        try:
            fr = f(x_right)
        except:
            fr = None

        if fr is not None and math.isfinite(fr):
            abs_fr = abs(fr)
            if abs_fr < best_abs:
                best_abs = abs_fr
                best_x = x_right
                best_f = fr
            if abs_fr <= 1e-8:
                return x_right
            if fa * fr < 0:
                lo, hi = x, x_right
                flo, fhi = fa, fr
                bracket_found = True
                break

        # Check left side
        x_left = x - d
        try:
            fl = f(x_left)
        except:
            fl = None

        if fl is not None and math.isfinite(fl):
            abs_fl = abs(fl)
            if abs_fl < best_abs:
                best_abs = abs_fl
                best_x = x_left
                best_f = fl
            if abs_fl <= 1e-8:
                return x_left
            if fa * fl < 0:
                lo, hi = x_left, x
                flo, fhi = fl, fa
                bracket_found = True
                break

        if time.time() > deadline:
            break

    if bracket_found:
        # Use Brent's method for fast, robust convergence
        try:
            root = brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=100)
            if time.time() < deadline:
                f_root = f(root)
                if math.isfinite(f_root) and abs(f_root) <= 1e-8:
                    return root
        except Exception:
            pass

        # Fallback: manual bisection
        a_b, b_b = min(lo, hi), max(lo, hi)
        try:
            fa_b = f(a_b)
            fb_b = f(b_b)
        except:
            fa_b, fb_b = None, None

        if fa_b is not None and fb_b is not None and math.isfinite(fa_b) and math.isfinite(fb_b):
            if fa_b * fb_b < 0:
                for _ in range(60):
                    if time.time() > deadline:
                        break
                    m = (a_b + b_b) / 2.0
                    try:
                        fm = f(m)
                    except:
                        break
                    if not math.isfinite(fm):
                        break
                    if abs(fm) <= 1e-8:
                        return m
                    if fa_b * fm < 0:
                        b_b = m
                        fb_b = fm
                    else:
                        a_b = m
                        fa_b = fm
                m = (a_b + b_b) / 2.0
                return m

    # If no bracket found, try to refine best_x using a few Newton-like steps
    # with careful derivative estimation.
    if best_abs > 1e-8:
        h = 1e-5 * max(1.0, abs(best_x))
        try:
            for _ in range(3):
                if time.time() > deadline:
                    break
                fx_best = f(best_x)
                if not math.isfinite(fx_best):
                    break
                fx_plus = f(best_x + h)
                fx_minus = f(best_x - h)
                if not (math.isfinite(fx_plus) and math.isfinite(fx_minus)):
                    break
                d = (fx_plus - fx_minus) / (2 * h)
                if d == 0 or not math.isfinite(d):
                    break
                x_new = best_x - fx_best / d
                if not math.isfinite(x_new):
                    break
                f_new = f(x_new)
                if not math.isfinite(f_new):
                    break
                if abs(f_new) < best_abs:
                    best_x = x_new
                    best_f = f_new
                    best_abs = abs(f_new)
                    if best_abs <= 1e-8:
                        break
        except:
            pass

    return best_x
# EVOLVE-BLOCK-END
