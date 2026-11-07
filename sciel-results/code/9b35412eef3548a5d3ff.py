import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0
    
    x = float(x0)
    
    # 1. Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
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

    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)

    # 2. Initialize second point
    h = max(1e-4, abs(x) * 1e-4)
    x_prev = x
    f_prev = fx
    
    x_curr = x + h
    try:
        f_curr = f(x_curr)
    except Exception:
        x_curr = x - h
        try:
            f_curr = f(x_curr)
        except Exception:
            return best_x

    if not math.isfinite(f_curr):
        x_curr = x - h
        try:
            f_curr = f(x_curr)
        except Exception:
            return best_x
        if not math.isfinite(f_curr):
            return best_x

    if abs(f_curr) <= 1e-8:
        return x_curr

    if abs(f_curr) < best_abs_fx:
        best_x = x_curr
        best_fx = f_curr
        best_abs_fx = abs(f_curr)

    history = [(x_prev, f_prev), (x_curr, f_curr)]
    
    max_iter = 80
    tol = 1e-10  # Tighter internal tolerance for polish
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # 1. Calculate Secant step
        x_secant = None
        if abs(f_curr - f_prev) > 1e-15:
            x_secant = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        # 2. Calculate IQI step using standard formula
        x_iqi = None
        if len(history) >= 3:
            p0, f0 = history[-3]
            p1, f1 = history[-2]
            p2, f2 = history[-1]
            
            # Check for distinctness in x values
            if (abs(p0 - p1) > 1e-14 and abs(p1 - p2) > 1e-14 and abs(p0 - p2) > 1e-14):
                try:
                    # Standard IQI: fit quadratic through (p0,f0), (p1,f1), (p2,f2)
                    # and find where it crosses zero
                    # Using Lagrange interpolation for the quadratic
                    # q(x) = f0 * (x-p1)(x-p2)/((p0-p1)(p0-p2)) 
                    #      + f1 * (x-p0)(x-p2)/((p1-p0)(p1-p2)) 
                    #      + f2 * (x-p0)(x-p1)/((p2-p0)(p2-p1))
                    # Set q(x) = 0 and solve for x
                    # This gives: x = [f0*p1*p2/((p0-p1)(p0-p2)) + f1*p0*p2/((p1-p0)(p1-p2)) + f2*p0*p1/((p2-p0)(p2-p1))]
                    #             / [f0/((p0-p1)(p0-p2)) + f1/((p1-p0)(p1-p2)) + f2/((p2-p0)(p2-p1))]
                    
                    denom0 = (p0 - p1) * (p0 - p2)
                    denom1 = (p1 - p0) * (p1 - p2)
                    denom2 = (p2 - p0) * (p2 - p1)
                    
                    if abs(denom0) > 1e-15 and abs(denom1) > 1e-15 and abs(denom2) > 1e-15:
                        num = f0 * p1 * p2 / denom0 + f1 * p0 * p2 / denom1 + f2 * p0 * p1 / denom2
                        den = f0 / denom0 + f1 / denom1 + f2 / denom2
                        
                        if abs(den) > 1e-15:
                            x_iqi = num / den
                except (ZeroDivisionError, OverflowError, FloatingPointError):
                    x_iqi = None
        
        # 3. Choose next step with safeguards
        x_next = None
        
        if x_iqi is not None and math.isfinite(x_iqi):
            # Use IQI if it's reasonable
            if x_secant is not None and math.isfinite(x_secant):
                dist_secant = abs(x_secant - x_curr)
                dist_iqi = abs(x_iqi - x_curr)
                # Prefer IQI if it's not too far from current point and secant
                if dist_iqi < 3.0 * dist_secant + 1e-6:
                    x_next = x_iqi
                else:
                    x_next = x_secant
            else:
                x_next = x_iqi
        elif x_secant is not None and math.isfinite(x_secant):
            x_next = x_secant
        else:
            x_next = x_curr + 1e-5 * (1 if f_curr > 0 else -1)
            
        # Safeguard: Limit step size
        max_step = max(1.0, abs(x_curr) * 0.3)
        if abs(x_next - x_curr) > max_step:
            x_next = x_curr + math.copysign(max_step, x_next - x_curr)

        # Evaluate f(x_next)
        try:
            f_next = f(x_next)
        except Exception:
            x_next = x_curr + 0.5 * (x_secant - x_curr) if x_secant else x_curr + 1e-6
            try:
                f_next = f(x_next)
            except Exception:
                break
            if not math.isfinite(f_next):
                break

        if not math.isfinite(f_next):
            break

        if abs(f_next) < best_abs_fx:
            best_x = x_next
            best_fx = f_next
            best_abs_fx = abs(f_next)

        if abs(f_next) <= tol:
            return x_next

        # Update history - keep only 3 most recent distinct points
        new_point = (x_next, f_next)
        history.append(new_point)
        if len(history) > 3:
            history.pop(0)
        
        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f_next
        
    # Final polish: try a few more iterations with tighter tolerance if we're close
    if best_abs_fx < 1e-6 and time.time() < deadline - 5.0:
        x_polish = best_x
        f_polish = best_fx
        for _ in range(10):
            if time.time() > deadline:
                break
            # Simple secant step for polish
            if abs(f_polish - f_prev) > 1e-15:
                x_new = x_polish - f_polish * (x_polish - x_prev) / (f_polish - f_prev)
            else:
                break
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
            if abs(f_new) < abs(f_polish):
                x_prev, f_prev = x_polish, f_polish
                x_polish, f_polish = x_new, f_new
                if abs(f_polish) <= 1e-8:
                    return x_polish
            else:
                break
    
    return best_x
