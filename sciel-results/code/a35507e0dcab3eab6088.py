"""Root finding for ill-conditioned equations: Robust Adaptive Hybrid."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin below 120s limit
    MAX_EVALS = 450   # Safety margin below 500 limit

    eval_count = 0
    best_x = float(x0)
    best_f = None

    def check_budget():
        nonlocal eval_count
        if eval_count > MAX_EVALS:
            return True
        if time.time() - start_time > DEADLINE:
            return True
        return False

    def safe_eval(x):
        nonlocal eval_count, best_x, best_f
        if check_budget():
            return None
        try:
            val = f(x)
            eval_count += 1
            if val is not None and np.isfinite(val):
                if best_f is None or abs(val) < abs(best_f):
                    best_f = val
                    best_x = x
                return val
        except Exception:
            pass
        return None

    def update_best(x, fx):
        nonlocal best_x, best_f
        if fx is not None and np.isfinite(fx):
            if best_f is None or abs(fx) < abs(best_f):
                best_f = fx
                best_x = x

    # First evaluate at x0
    fx0 = safe_eval(x0)
    if fx0 is None:
        return best_x
    update_best(x0, fx0)
    if abs(fx0) <= 1e-8:
        return best_x

    # Try to establish a bracket using adaptive steps
    a = float(x0)
    b = float(x0)
    fa = fx0
    fb = None

    # Try to find a bracket by expanding outward
    step = max(1.0, abs(a)) * 1e-3
    for i in range(50):
        if check_budget():
            break
        # Try to the right
        if fb is None or (fb > 0 and fa > 0):
            x_try = a + step
            f_try = safe_eval(x_try)
            if f_try is not None and np.isfinite(f_try):
                update_best(x_try, f_try)
                if fa * f_try < 0:
                    b = x_try
                    fb = f_try
                    break
                elif abs(f_try) < abs(fa):
                    # Move in this direction
                    a = x_try
                    fa = f_try
                    step *= 2
                else:
                    step *= 2
                    continue
        # Try to the left
        if fb is None or (fb < 0 and fa < 0):
            x_try = a - step
            f_try = safe_eval(x_try)
            if f_try is not None and np.isfinite(f_try):
                update_best(x_try, f_try)
                if fa * f_try < 0:
                    b = x_try
                    fb = f_try
                    break
                elif abs(f_try) < abs(fa):
                    a = x_try
                    fa = f_try
                    step *= 2
                else:
                    step *= 2
                    continue
        step *= 1.5
        if step > 1e10:
            break

    # If we have a bracket, use bisection + Newton hybrid
    if fb is not None and np.isfinite(fb) and fa * fb < 0:
        lo, hi = min(a, b), max(a, b)
        flo, fhi = fa, fb
        if abs(flo) < abs(fhi):
            lo, hi = hi, lo
            flo, fhi = fhi, flo
        # Ensure lo has smaller |f|
        if abs(flo) > abs(fhi):
            lo, hi = hi, lo
            flo, fhi = fhi, flo

        x = lo
        fx = flo
        for _ in range(80):
            if check_budget():
                break
            if abs(fx) <= 1e-8:
                return x
            # Try Newton step
            h = 1e-7 * max(1.0, abs(x))
            f_xh = safe_eval(x + h)
            if f_xh is not None and np.isfinite(f_xh):
                d = (f_xh - fx) / h
                if d != 0 and np.isfinite(d):
                    x_new = x - fx / d
                    # Check if Newton step stays in bracket
                    if lo < x_new < hi:
                        f_new = safe_eval(x_new)
                        if f_new is not None and np.isfinite(f_new):
                            update_best(x_new, f_new)
                            x = x_new
                            fx = f_new
                            # Update bracket
                            if fx * flo < 0:
                                hi = x
                                fhi = fx
                            else:
                                lo = x
                                flo = fx
                            continue
            # Bisection fallback
            x_mid = (lo + hi) / 2.0
            f_mid = safe_eval(x_mid)
            if f_mid is None or not np.isfinite(f_mid):
                break
            update_best(x_mid, f_mid)
            if f_mid * flo < 0:
                hi = x_mid
                fhi = f_mid
            else:
                lo = x_mid
                flo = f_mid
            x = x_mid
            fx = f_mid
        return best_x

    # No bracket found: use secant/Newton with damping
    x1 = float(x0)
    fx1 = fx0
    x2 = x1 + max(1.0, abs(x1)) * 1e-4
    fx2 = safe_eval(x2)
    if fx2 is None or not np.isfinite(fx2):
        return best_x
    update_best(x2, fx2)

    for _ in range(100):
        if check_budget():
            break
        if abs(fx1) <= 1e-8:
            return x1
        # Secant step
        if fx2 != fx1:
            x_new = x1 - fx1 * (x1 - x2) / (fx1 - fx2)
        else:
            # Use finite difference derivative
            h = 1e-6 * max(1.0, abs(x1))
            f_xh = safe_eval(x1 + h)
            if f_xh is None or not np.isfinite(f_xh):
                break
            d = (f_xh - fx1) / h
            if d == 0 or not np.isfinite(d):
                break
            x_new = x1 - fx1 / d

        if not np.isfinite(x_new):
            break

        # Damping: ensure step is reasonable
        step_size = abs(x_new - x1)
        max_step = max(1.0, abs(x1)) * 10.0
        if step_size > max_step:
            x_new = x1 + math.copysign(max_step, x_new - x1)

        f_new = safe_eval(x_new)
        if f_new is None or not np.isfinite(f_new):
            break
        update_best(x_new, f_new)

        x2 = x1
        fx2 = fx1
        x1 = x_new
        fx1 = f_new

    return best_x

# EVOLVE-BLOCK-END
