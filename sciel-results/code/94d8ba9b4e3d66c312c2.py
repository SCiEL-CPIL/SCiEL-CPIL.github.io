"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    def time_ok():
        return time.time() < deadline
    
    x = float(x0)
    
    # 1. Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # If non-finite, try to find a nearby finite point
        found = False
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                except Exception:
                    continue
                if math.isfinite(fx_test):
                    x = x_test
                    fx = fx_test
                    found = True
                    break
            if found:
                break
        if not found:
            return x0
        if abs(fx) <= 1e-8:
            return x

    if abs(fx) <= 1e-8:
        return x

    # Track best point found
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)
    
    # Phase 1: Safeguarded Newton-Raphson
    # Approximate derivative using central difference
    # x_new = x - f(x) / f'(x)
    # Safeguard: if step is too large or increases |f|, reduce step
    
    # Initial derivative approximation
    h = 1e-8
    try:
        f_plus = f(x + h)
        f_minus = f(x - h)
        if math.isfinite(f_plus) and math.isfinite(f_minus):
            fp = (f_plus - f_minus) / (2 * h)
        else:
            fp = None
    except Exception:
        fp = None
    
    if fp is None or abs(fp) < 1e-15:
        # Try forward difference
        try:
            f_plus = f(x + h)
            if math.isfinite(f_plus):
                fp = (f_plus - fx) / h
            else:
                fp = None
        except Exception:
            fp = None
    
    if fp is None or abs(fp) < 1e-15:
        # Fallback to secant method if derivative is problematic
        # Get a second point for secant
        x2 = x + 1e-3
        try:
            f2 = f(x2)
        except Exception:
            x2 = x - 1e-3
            try:
                f2 = f(x2)
            except Exception:
                return best_x
        if not math.isfinite(f2):
            for dx in [1e-4, 1e-5, 1e-2, 0.1, 1.0]:
                for sign in [1, -1]:
                    x_try = x + sign * dx
                    try:
                        f_try = f(x_try)
                    except Exception:
                        continue
                    if math.isfinite(f_try):
                        x2, f2 = x_try, f_try
                        break
                if math.isfinite(f2):
                    break
        if not math.isfinite(f2):
            return best_x
        if abs(f2) < best_abs_fx:
            best_x = x2
            best_fx = f2
            best_abs_fx = abs(f2)
        if abs(f2) <= 1e-8:
            return x2
        
        # Secant method
        x1, f1 = x, fx
        for i in range(80):
            if not time_ok():
                break
            if abs(f1) <= 1e-8:
                return x1
            if abs(f2) <= 1e-8:
                return x2
            if abs(f2 - f1) < 1e-15:
                x_new = x2 + 1e-4
            else:
                x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
            max_step = max(1.0, abs(x2) * 0.5)
            if abs(x_new - x2) > max_step:
                x_new = x2 + math.copysign(max_step, x_new - x2)
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                x_new = 0.5 * (x2 + x1)
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(f_new):
                    break
            if abs(f_new) < best_abs_fx:
                best_x = x_new
                best_fx = f_new
                best_abs_fx = abs(f_new)
            if abs(f_new) <= 1e-8:
                return x_new
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
        return best_x
    
    # Newton-Raphson with safeguards
    # Use a damping factor to prevent overshoot
    damping = 1.0
    
    for i in range(100):
        if not time_ok():
            break
        
        # Compute Newton step
        step = fx / fp
        x_new = x - step
        
        # Safeguard: if step is too large, dampen it
        max_step = max(1.0, abs(x) * 0.5)
        if abs(step) > max_step:
            step = math.copysign(max_step, step)
            x_new = x - step
        
        # Try the Newton step
        try:
            f_new = f(x_new)
        except Exception:
            # If Newton step fails, try a smaller step
            x_new = x - 0.5 * step
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
        
        if not math.isfinite(f_new):
            # Try even smaller step
            x_new = x - 0.25 * step
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
        
        if abs(f_new) < best_abs_fx:
            best_x = x_new
            best_fx = f_new
            best_abs_fx = abs(f_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
        
        # Check if we made progress
        if abs(f_new) < abs(fx) * 0.9:
            # Good progress, accept the step
            x, fx = x_new, f_new
            # Recompute derivative
            try:
                f_plus = f(x + h)
                f_minus = f(x - h)
                if math.isfinite(f_plus) and math.isfinite(f_minus):
                    fp = (f_plus - f_minus) / (2 * h)
                else:
                    fp = None
            except Exception:
                fp = None
            if fp is None or abs(fp) < 1e-15:
                try:
                    f_plus = f(x + h)
                    if math.isfinite(f_plus):
                        fp = (f_plus - fx) / h
                    else:
                        fp = None
                except Exception:
                    fp = None
            if fp is None or abs(fp) < 1e-15:
                # Derivative failed, fall back to secant-like step
                # Use previous two points
                break
        else:
            # No good progress, try to find a bracket or use bisection
            # First, try to find a sign change nearby
            found_bracket = False
            for dx in [abs(step) * 0.5, abs(step) * 0.25, abs(step) * 0.1, 1e-3, 1e-2, 0.1, 1.0]:
                for sign in [1, -1]:
                    x_test = x + sign * dx
                    try:
                        f_test = f(x_test)
                    except Exception:
                        continue
                    if not math.isfinite(f_test):
                        continue
                    if abs(f_test) < best_abs_fx:
                        best_x = x_test
                        best_fx = f_test
                        best_abs_fx = abs(f_test)
                    if abs(f_test) <= 1e-8:
                        return x_test
                    if fx * f_test < 0:
                        # Found a bracket, use bisection
                        lo, hi = (x, x_test) if sign == 1 else (x_test, x)
                        flo, fhi = (fx, f_test) if sign == 1 else (f_test, fx)
                        for j in range(50):
                            if not time_ok():
                                break
                            if abs(flo) <= 1e-8:
                                return lo
                            if abs(fhi) <= 1e-8:
                                return hi
                            mid = 0.5 * (lo + hi)
                            try:
                                fmid = f(mid)
                            except Exception:
                                break
                            if not math.isfinite(fmid):
                                break
                            if abs(fmid) < best_abs_fx:
                                best_x = mid
                                best_fx = fmid
                                best_abs_fx = abs(fmid)
                            if abs(fmid) <= 1e-8:
                                return mid
                            if flo * fmid <= 0:
                                hi, fhi = mid, fmid
                            else:
                                lo, flo = mid, fmid
                        if abs(best_fx) <= 1e-8:
                            return best_x
                        return best_x
            # No bracket found, take a small step in the direction that decreases |f|
            for dx in [1e-3, 1e-2, 0.1, 1.0]:
                for sign in [1, -1]:
                    x_test = x + sign * dx
                    try:
                        f_test = f(x_test)
                    except Exception:
                        continue
                    if not math.isfinite(f_test):
                        continue
                    if abs(f_test) < abs(fx):
                        x, fx = x_test, f_test
                        if abs(fx) < best_abs_fx:
                            best_x = x
                            best_fx = fx
                            best_abs_fx = abs(fx)
                        if abs(fx) <= 1e-8:
                            return x
                        found_bracket = True
                        break
                if found_bracket:
                    break
            if not found_bracket:
                break
    
    return best_x
# EVOLVE-BLOCK-END
