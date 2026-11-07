"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """Find root of f near x0 using IQI with Newton and bisection fallback."""
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    x = float(x0)
    fx = f(x)
    best_x = x
    best_fx = abs(fx)
    
    # Keep history for IQI: list of (x, fx)
    history = [(x, fx)]
    
    # Phase 1: Try inverse quadratic interpolation (IQI) for up to 30 iterations
    for i in range(30):
        if time.time() > deadline:
            break
        if abs(fx) <= 1e-12:
            return x
        if abs(fx) < best_fx:
            best_fx = abs(fx)
            best_x = x
        
        # Need at least 3 points for IQI
        if len(history) < 3:
            # Use Newton to generate more points
            h = 1e-6 * max(1.0, abs(x))
            x_new = x + h
            fx_new = f(x_new)
            if not math.isfinite(fx_new):
                h = -h
                x_new = x + h
                fx_new = f(x_new)
            if math.isfinite(fx_new):
                history.append((x_new, fx_new))
                if abs(fx_new) < best_fx:
                    best_fx = abs(fx_new)
                    best_x = x_new
            continue
        
        # Take last 3 distinct points
        pts = history[-3:]
        x1, f1 = pts[0]
        x2, f2 = pts[1]
        x3, f3 = pts[2]
        
        # Check for distinct x values
        if abs(x1 - x2) < 1e-15 or abs(x2 - x3) < 1e-15 or abs(x1 - x3) < 1e-15:
            # Fall back to Newton step
            h = 1e-7 * max(1.0, abs(x))
            fp = f(x + h)
            fm = f(x - h)
            d = (fp - fm) / (2.0 * h)
            if d == 0 or not math.isfinite(d):
                break
            step = fx / d
            max_step = 10.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = max_step * math.copysign(1.0, step)
            x_new = x - step
            if not math.isfinite(x_new):
                break
            fx_new = f(x_new)
            if not math.isfinite(fx_new):
                break
            history.append((x_new, fx_new))
            if abs(fx_new) < best_fx:
                best_fx = abs(fx_new)
                best_x = x_new
            x = x_new
            fx = fx_new
            continue
        
        # Inverse quadratic interpolation: fit quadratic q(y) such that q(f_i) = x_i
        # We want to find y such that q(y) is the root, i.e., solve for x where f(x)=0
        # IQI: x_new = x3 + (x3-x1)*(x3-x2) / ((x2-x1)*f3/(f3-f1) + (x3-x2)*f3/(f3-f2)) * ... 
        # Standard formula:
        # Let d1 = (f2 - f1), d2 = (f3 - f2)
        # x_new = x3 + (x3-x1)*(x3-x2) / (d1*(x3-x2)*f3/(f3-f1) + d2*(x3-x1)*f3/(f3-f2))... 
        # Simpler form:
        # x_new = x3 - f3 * (x3-x1)*(x3-x2) / ((f3-f1)*(f3-f2)) * (1/(x3-x2) - 1/(x3-x1))... 
        # Let me use the standard IQI formula:
        # Given three points (x1,f1), (x2,f2), (x3,f3), the IQI step is:
        # x_new = x3 + (x3-x1)*(x3-x2) / ((x2-x1)*f3/(f3-f1) + (x3-x2)*f3/(f3-f2))
        # Actually, the correct formula for inverse quadratic interpolation:
        # We fit a quadratic p(y) = a*y^2 + b*y + c such that p(f1)=x1, p(f2)=x2, p(f3)=x3
        # Then x_new = p(0)
        
        # Using Lagrange interpolation:
        # p(0) = f1*f2/(f1-f2) * (x1 - x3) / (f1 - f3) ... this is getting complex
        # Let me use the standard numerical formula:
        
        # x_new = x3 - f3 * (x3-x1)*(x3-x2) / ((f3-f1)*(f3-f2))
        # Wait, that's not quite right either. Let me derive properly.
        
        # The inverse quadratic through (f1,x1), (f2,x2), (f3,x3) evaluated at y=0:
        # x_new = f1*f2/(f1-f2) * (x1-x3)/(f1-f3) ... no
        
        # Standard formula from numerical analysis:
        # x_new = x3 + (x3-x1)*(x3-x2) / ((x2-x1)*f3/(f3-f1) + (x3-x2)*f3/(f3-f2))
        
        denom = (x2 - x1) * f3 / (f3 - f1) + (x3 - x2) * f3 / (f3 - f2)
        if abs(denom) < 1e-30:
            # Degenerate, use Newton
            h = 1e-7 * max(1.0, abs(x))
            fp = f(x + h)
            fm = f(x - h)
            d = (fp - fm) / (2.0 * h)
            if d == 0 or not math.isfinite(d):
                break
            step = fx / d
            max_step = 10.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = max_step * math.copysign(1.0, step)
            x_new = x - step
            if not math.isfinite(x_new):
                break
            fx_new = f(x_new)
            if not math.isfinite(fx_new):
                break
            history.append((x_new, fx_new))
            if abs(fx_new) < best_fx:
                best_fx = abs(fx_new)
                best_x = x_new
            x = x_new
            fx = fx_new
            continue
        
        x_new = x3 + (x3 - x1) * (x3 - x2) / denom
        
        # Limit step
        max_step = 10.0 * max(1.0, abs(x))
        if abs(x_new - x) > max_step:
            x_new = x + max_step * math.copysign(1.0, x_new - x)
        
        if not math.isfinite(x_new):
            break
        
        fx_new = f(x_new)
        if not math.isfinite(fx_new):
            break
        
        history.append((x_new, fx_new))
        # Keep only last 10 points
        if len(history) > 10:
            history = history[-10:]
        
        if abs(fx_new) < best_fx:
            best_fx = abs(fx_new)
            best_x = x_new
        
        x = x_new
        fx = fx_new
    
    if abs(fx) < best_fx:
        best_x = x
        best_fx = abs(fx)
    
    if best_fx <= 1e-8:
        return best_x
    
    # Phase 2: Damped Newton fallback
    for i in range(30):
        if time.time() > deadline:
            break
        if abs(fx) <= 1e-12:
            return x
        if abs(fx) < best_fx:
            best_fx = abs(fx)
            best_x = x
        
        h = 1e-7 * max(1.0, abs(x))
        fp = f(x + h)
        fm = f(x - h)
        d = (fp - fm) / (2.0 * h)
        
        if d == 0 or not math.isfinite(d):
            break
        
        step = fx / d
        max_step = 10.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = max_step * math.copysign(1.0, step)
        
        x_new = x - step
        
        if not math.isfinite(x_new):
            break
        
        fx_new = f(x_new)
        if not math.isfinite(fx_new):
            break
        
        if abs(fx_new) < abs(fx):
            x = x_new
            fx = fx_new
        else:
            x_new = x - 0.5 * step
            fx_new = f(x_new)
            if abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
            else:
                x_new = x - 0.25 * step
                fx_new = f(x_new)
                if abs(fx_new) < abs(fx):
                    x = x_new
                    fx = fx_new
                else:
                    break
    
    if abs(fx) < best_fx:
        best_x = x
        best_fx = abs(fx)
    
    if best_fx <= 1e-8:
        return best_x
    
    # Phase 3: Bisection fallback
    a = best_x - 1.0
    b = best_x + 1.0
    fa = f(a)
    fb = f(b)
    
    for _ in range(20):
        if time.time() > deadline:
            break
        if not math.isfinite(fa) or not math.isfinite(fb):
            break
        if fa * fb <= 0:
            break
        a = best_x - (b - a)
        b = best_x + (b - a)
        fa = f(a)
        fb = f(b)
    
    if math.isfinite(fa) and math.isfinite(fb) and fa * fb <= 0:
        for _ in range(60):
            if time.time() > deadline:
                break
            mid = 0.5 * (a + b)
            fm = f(mid)
            if abs(fm) <= 1e-12:
                return mid
            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
            if abs(b - a) < 1e-14:
                break
        return 0.5 * (a + b)
    
    return best_x
# EVOLVE-BLOCK-END
