import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    x = float(x0)
    
    # 1. Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # Try to find a nearby finite point
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

    # Best point tracking
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)

    # 2. Initialize second point for Secant
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
        # Try another direction
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

    # History for IQI: (x, f)
    # We need 3 points. We have (x_prev, f_prev) and (x_curr, f_curr).
    # Let's order them so x_curr is the most recent.
    # Actually, for IQI, the order matters for stability. We use the last 3 distinct points.
    # Let's maintain a list of (x, f)
    history = [(x_prev, f_prev), (x_curr, f_curr)]
    
    # Ensure x_curr is the "current" best guess for the next step
    # Usually we step from the most recent point.
    # Let's set x = x_curr, fx = f_curr for the loop start?
    # Or keep x as the one we are stepping from.
    # Standard Secant: x_{n+1} = x_n - f_n * (x_n - x_{n-1}) / (f_n - f_{n-1})
    # We will update history and pick the next x.
    
    x = x_curr
    fx = f_curr
    x_prev = x_prev
    f_prev = f_prev

    max_iter = 100
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # 1. Calculate Secant step
        x_secant = None
        if abs(f_curr - f_prev) > 1e-14:
            x_secant = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        # 2. Calculate IQI step if we have 3 points
        x_iqi = None
        if len(history) >= 3:
            # Get last 3 distinct points
            # Ensure points are distinct enough
            p0, f0 = history[-3]
            p1, f1 = history[-2]
            p2, f2 = history[-1]
            
            # Check for distinctness
            if abs(f0 - f1) > 1e-14 and abs(f1 - f2) > 1e-14 and abs(f0 - f2) > 1e-14:
                try:
                    # x_new = f0*f1*f2 / ((f0-f1)(f0-f2)) ... wait, the formula derived earlier:
                    # x_new = f0 * (f1*f2)/((f0-f1)*(f0-f2)) + f1 * (f0*f2)/((f1-f0)*(f1-f2)) + f2 * (f0*f1)/((f2-f0)*(f2-f1))
                    
                    term0 = f0 * (f1 * f2) / ((f0 - f1) * (f0 - f2))
                    term1 = f1 * (f0 * f2) / ((f1 - f0) * (f1 - f2))
                    term2 = f2 * (f0 * f1) / ((f2 - f0) * (f2 - f1))
                    
                    x_iqi = term0 + term1 + term2
                except (ZeroDivisionError, OverflowError):
                    x_iqi = None
        
        # 3. Choose next step
        x_next = None
        
        # Prefer IQI if it's finite and not too far from Secant
        if x_iqi is not None and math.isfinite(x_iqi):
            # Safeguard: If IQI step is much larger than Secant, it might be unstable
            # Use a heuristic: if |x_iqi - x_curr| < 2 * |x_secant - x_curr| (if secant exists)
            if x_secant is not None and math.isfinite(x_secant):
                dist_secant = abs(x_secant - x_curr)
                dist_iqi = abs(x_iqi - x_curr)
                if dist_iqi < 2.0 * dist_secant or dist_iqi < 1e-3:
                    x_next = x_iqi
                else:
                    x_next = x_secant
            else:
                x_next = x_iqi
        elif x_secant is not None and math.isfinite(x_secant):
            x_next = x_secant
        else:
            # Fallback: small step
            x_next = x_curr + 1e-4
            
        # Safeguard: Limit step size to prevent running away
        # If the step is huge, shrink it
        max_step = max(1.0, abs(x_curr) * 0.5)
        if abs(x_next - x_curr) > max_step:
            x_next = x_curr + math.copysign(max_step, x_next - x_curr)

        # Evaluate f(x_next)
        try:
            f_next = f(x_next)
        except Exception:
            # If evaluation fails, try a smaller step
            x_next = x_curr + 0.5 * (x_secant - x_curr) if x_secant else x_curr + 1e-5
            try:
                f_next = f(x_next)
            except Exception:
                break
            if not math.isfinite(f_next):
                break

        if not math.isfinite(f_next):
            # Try moving towards a previously good point or just break
            break

        # Update best
        if abs(f_next) < best_abs_fx:
            best_x = x_next
            best_fx = f_next
            best_abs_fx = abs(f_next)

        if abs(f_next) <= 1e-8:
            return x_next

        # Update history
        # Remove oldest if we have more than 3
        history.append((x_next, f_next))
        if len(history) > 3:
            history.pop(0)
        
        # Shift points for next iteration
        # The "current" point for the next Secant/IQI calculation should be the new one
        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f_next
        
        # Optional: If we didn't make progress, we might want to try a different direction
        # But Secant/IQI usually handle this if the function is smooth.
        
    return best_x
