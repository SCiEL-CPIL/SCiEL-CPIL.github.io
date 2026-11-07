import math
import time
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0

    x = float(x0)
    try:
        fx = f(x)
    except:
        return x

    if not math.isfinite(fx):
        return x

    best_x = x
    best_f = fx
    best_abs = abs(fx)

    if best_abs <= 1e-8:
        return x

    fa = fx

    # Estimate local derivative to choose initial step size
    h_est = 1e-5 * max(1.0, abs(x))
    try:
        fx_plus = f(x + h_est)
        fx_minus = f(x - h_est)
        if math.isfinite(fx_plus) and math.isfinite(fx_minus):
            slope = (fx_plus - fx_minus) / (2 * h_est)
            abs_slope = abs(slope)
        else:
            abs_slope = 1.0
    except:
        abs_slope = 1.0

    # If slope is steep, use smaller steps. If flat, larger.
    # Base step size logic:
    # We want to catch oscillations. Assume typical oscillation period is related to 1/slope?
    # If slope is 100, period might be 0.1. Step should be < 0.1.
    # If slope is 0.01, period might be 100. Step can be 10.
    
    # Let's define a "characteristic scale"
    # scale = 1 / max(abs_slope, 1e-6)
    # But we also need to explore far.
    
    # Strategy:
    # Start with step = min(1.0, 0.1 / max(abs_slope, 1e-6))
    # But cap it.
    
    initial_step = 1.0
    if abs_slope > 1e-3:
        initial_step = min(1.0, 0.5 / abs_slope)
        # Ensure initial_step is not too small (wasteful)
        initial_step = max(initial_step, 1e-4)
    else:
        initial_step = 1.0

    # Generate deltas
    # We will use a geometric progression starting from initial_step
    # Ratio 2.0 is a good balance.
    
    deltas = []
    d = initial_step
    while d <= 1e6:
        deltas.append(d)
        d *= 2.0
        
    # Also add some very small steps just in case initial_step was too big
    # and some very large steps
    small_deltas = [1e-6, 1e-5, 1e-4]
    for sd in small_deltas:
        if sd < initial_step:
            deltas.insert(0, sd)
            
    # Remove duplicates and sort
    deltas = sorted(set(deltas))

    prev_x_right = x
    prev_f_right = fx
    prev_x_left = x
    prev_f_left = fx

    bracket_found = False
    bracket_lo, bracket_hi = 0, 0
    bracket_flo, bracket_fhi = 0, 0

    for d in deltas:
        if time.time() > deadline:
            break

        # Right side
        x_r = x + d
        try:
            f_r = f(x_r)
        except:
            f_r = None

        if f_r is not None and math.isfinite(f_r):
            abs_f_r = abs(f_r)
            if abs_f_r < best_abs:
                best_abs = abs_f_r
                best_x = x_r
                best_f = f_r

            if abs_f_r <= 1e-8:
                return x_r

            if prev_f_right is not None and math.isfinite(prev_f_right):
                if prev_f_right * f_r < 0:
                    bracket_found = True
                    bracket_lo = prev_x_right
                    bracket_hi = x_r
                    bracket_flo = prev_f_right
                    bracket_fhi = f_r
                    break

            prev_x_right = x_r
            prev_f_right = f_r

        # Left side
        x_l = x - d
        try:
            f_l = f(x_l)
        except:
            f_l = None

        if f_l is not None and math.isfinite(f_l):
            abs_f_l = abs(f_l)
            if abs_f_l < best_abs:
                best_abs = abs_f_l
                best_x = x_l
                best_f = f_l

            if abs_f_l <= 1e-8:
                return x_l

            if prev_f_left is not None and math.isfinite(prev_f_left):
                if prev_f_left * f_l < 0:
                    bracket_found = True
                    bracket_lo = x_l
                    bracket_hi = prev_x_left
                    bracket_flo = f_l
                    bracket_fhi = prev_f_left
                    break

            prev_x_left = x_l
            prev_f_left = f_l

    if bracket_found:
        try:
            root = brentq(f, bracket_lo, bracket_hi, xtol=1e-12, rtol=1e-12, maxiter=50)
            if time.time() < deadline:
                f_root = f(root)
                if math.isfinite(f_root) and abs(f_root) <= 1e-8:
                    return root
        except Exception:
            pass
        
        # Fallback bisection
        a_b, b_b = min(bracket_lo, bracket_hi), max(bracket_lo, bracket_hi)
        fa_b, fb_b = bracket_flo, bracket_fhi
        for _ in range(50):
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
        return (a_b + b_b) / 2.0

    # If no bracket, refine best_x
    if best_abs > 1e-8 and time.time() < deadline - 5:
        h = 1e-5 * max(1.0, abs(best_x))
        x1 = best_x
        f1 = best_f
        x2 = best_x + h
        try:
            f2 = f(x2)
        except:
            f2 = None
        if f2 is not None and math.isfinite(f2):
            for _ in range(5):
                if time.time() > deadline:
                    break
                if abs(f2 - f1) < 1e-15:
                    break
                x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
                try:
                    f3 = f(x3)
                except:
                    break
                if not math.isfinite(f3):
                    break
                if abs(f3) < best_abs:
                    best_abs = abs(f3)
                    best_x = x3
                    best_f = f3
                    if best_abs <= 1e-8:
                        return x3
                x1, f1 = x2, f2
                x2, f2 = x3, f3

    return best_x
# EVOLVE-BLOCK-END
