import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0
    
    x = float(x0)
    fx = f(x)
    
    if abs(fx) <= 1e-8:
        return x
    
    # Best tracking
    best_x = x
    best_fx = fx
    best_abs = abs(fx)
    
    # Initial secant setup
    h = 1e-4 * max(1.0, abs(x))
    x2 = x + h
    fx2 = f(x2)
    
    if abs(fx2) <= 1e-8:
        return x2
    if abs(fx2) < best_abs:
        best_x, best_fx, best_abs = x2, fx2, abs(fx2)
    
    # Estimate derivative from secant
    if abs(x2 - x) > 1e-15:
        dfx = (fx2 - fx) / (x2 - x)
    else:
        dfx = 1.0
    
    # Phase 1: Secant for a few steps to get closer
    x1, f1 = x, fx
    x2, f2 = x2, fx2
    
    for i in range(5):
        if time.time() > deadline:
            return best_x
        
        if abs(f2) <= 1e-8:
            return x2
        if abs(f2) < best_abs:
            best_x, best_fx, best_abs = x2, f2, abs(f2)
        
        denom = f2 - f1
        if abs(denom) < 1e-15:
            # Nudge
            x2 = x2 + 1e-4 * max(1.0, abs(x2))
            f2 = f(x2)
            if abs(f2) <= 1e-8:
                return x2
            if abs(f2) < best_abs:
                best_x, best_fx, best_abs = x2, f2, abs(f2)
            continue
        
        x_new = x2 - f2 * (x2 - x1) / denom
        
        # Check for divergence
        if not math.isfinite(x_new):
            return best_x
        
        f_new = f(x_new)
        
        if not math.isfinite(f_new):
            return best_x
        
        if abs(f_new) <= 1e-8:
            return x_new
        if abs(f_new) < best_abs:
            best_x, best_fx, best_abs = x_new, f_new, abs(f_new)
        
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new
    
    # Phase 2: Newton with derivative estimation
    # Current best point is x2, f2
    xn = x2
    fn = f2
    
    # Re-estimate derivative at xn using central difference
    h_step = 1e-6 * max(1.0, abs(xn))
    xm = xn - h_step
    xp = xn + h_step
    fm = f(xm)
    fp = f(xp)
    
    if not math.isfinite(fm) or not math.isfinite(fp):
        return best_x
    
    dfn = (fp - fm) / (2.0 * h_step)
    
    if abs(dfn) < 1e-15:
        # Derivative too small, fall back to secant-like
        dfn = 1.0
    
    # Newton iterations
    for i in range(30):
        if time.time() > deadline:
            return best_x
        
        if abs(fn) <= 1e-8:
            return xn
        
        if abs(fn) < best_abs:
            best_x, best_fx, best_abs = xn, fn, abs(fn)
        
        # Newton step
        dx = fn / dfn
        x_new = xn - dx
        
        # Check step size
        if not math.isfinite(x_new):
            return best_x
        
        # If step is too large, dampen
        if abs(dx) > 10.0 * max(1.0, abs(xn)):
            x_new = xn - math.copysign(min(abs(dx), 10.0 * max(1.0, abs(xn))), dx)
        
        f_new = f(x_new)
        
        if not math.isfinite(f_new):
            return best_x
        
        if abs(f_new) <= 1e-8:
            return x_new
        if abs(f_new) < best_abs:
            best_x, best_fx, best_abs = x_new, f_new, abs(f_new)
        
        # If function value increased significantly, re-estimate derivative and dampen
        if abs(f_new) > abs(fn) * 1.5:
            # Re-estimate derivative at xn
            h_step = 1e-6 * max(1.0, abs(xn))
            xm = xn - h_step
            xp = xn + h_step
            fm = f(xm)
            fp = f(xp)
            if not math.isfinite(fm) or not math.isfinite(fp):
                return best_x
            dfn = (fp - fm) / (2.0 * h_step)
            if abs(dfn) < 1e-15:
                dfn = 1.0
            # Dampen step
            x_new = xn - 0.5 * fn / dfn
            f_new = f(x_new)
            if not math.isfinite(f_new):
                return best_x
            if abs(f_new) <= 1e-8:
                return x_new
            if abs(f_new) < best_abs:
                best_x, best_fx, best_abs = x_new, f_new, abs(f_new)
        
        # Update
        xn = x_new
        fn = f_new
        
        # Re-estimate derivative every 3 steps
        if i % 3 == 2:
            h_step = 1e-6 * max(1.0, abs(xn))
            xm = xn - h_step
            xp = xn + h_step
            fm = f(xm)
            fp = f(xp)
            if not math.isfinite(fm) or not math.isfinite(fp):
                break
            dfn = (fp - fm) / (2.0 * h_step)
            if abs(dfn) < 1e-15:
                break
    
    # Phase 3: Polish pass with very tight tolerance
    # Do a few more Newton steps if we're close
    if best_abs < 1e-6:
        xn = best_x
        fn = f(xn)
        if abs(fn) <= 1e-8:
            return xn
        
        h_step = 1e-8 * max(1.0, abs(xn))
        xm = xn - h_step
        xp = xn + h_step
        fm = f(xm)
        fp = f(xp)
        if math.isfinite(fm) and math.isfinite(fp):
            dfn = (fp - fm) / (2.0 * h_step)
            if abs(dfn) > 1e-15:
                for i in range(10):
                    if time.time() > deadline:
                        break
                    if abs(fn) <= 1e-8:
                        return xn
                    x_new = xn - fn / dfn
                    if not math.isfinite(x_new):
                        break
                    f_new = f(x_new)
                    if not math.isfinite(f_new):
                        break
                    if abs(f_new) <= 1e-8:
                        return x_new
                    if abs(f_new) < best_abs:
                        best_x, best_fx, best_abs = x_new, f_new, abs(f_new)
                    # Re-estimate derivative
                    h_step = 1e-8 * max(1.0, abs(x_new))
                    xm = x_new - h_step
                    xp = x_new + h_step
                    fm = f(xm)
                    fp = f(xp)
                    if not math.isfinite(fm) or not math.isfinite(fp):
                        break
                    dfn = (fp - fm) / (2.0 * h_step)
                    if abs(dfn) < 1e-15:
                        break
                    xn = x_new
                    fn = f_new
    
    return best_x
