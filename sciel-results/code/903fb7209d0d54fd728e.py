"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    deadline = time.time() + 138  # 138s safety margin (evaluator kills at 150s)
    eval_count = 0
    max_evals = 500
    best_x = float(x0)
    best_fval = float('inf')
    
    def safe_f(x):
        nonlocal eval_count, best_x, best_fval
        if time.time() > deadline:
            raise TimeoutError()
        eval_count += 1
        if eval_count > max_evals:
            raise ValueError("max evals")
        val = f(x)
        val = float(val)
        if math.isfinite(val):
            if abs(val) < abs(best_fval):
                best_fval = val
                best_x = x
        return val
    
    x = float(x0)
    
    # Get initial points
    f0 = safe_f(x)
    if abs(f0) <= 1e-8:
        return x
    
    # Get a second point for secant
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    f1 = safe_f(x1)
    if abs(f1) <= 1e-8:
        return x1
    
    # Get third point via secant
    if f1 != f0:
        x2 = x1 - f1 * (x1 - x) / (f1 - f0)
    else:
        x2 = x1 + h
    
    f2 = safe_f(x2)
    if abs(f2) <= 1e-8:
        return x2
    
    # Main loop: safeguarded inverse quadratic interpolation
    # Points are (x, f) pairs; we maintain the 3 best by |f|
    pts = [(x, f0), (x1, f1), (x2, f2)]
    
    for _ in range(495):  # We've used 3 evals, have 497 left
        if time.time() > deadline:
            break
        
        # Sort by |f|
        pts.sort(key=lambda p: abs(p[1]))
        
        # Check success
        if abs(pts[0][1]) <= 1e-8:
            return pts[0][0]
        
        xa, fa = pts[0]
        xb, fb = pts[1]
        xc, fc = pts[2]
        
        # Compute secant step using the two best points
        if abs(fa - fb) > 1e-30:
            x_secant = xb - fb * (xb - xa) / (fb - fa)
        else:
            x_secant = (xa + xb) / 2
        
        # Compute IQI step
        x_iqi = None
        try:
            # Use the standard Lagrange formula for inverse quadratic interpolation
            # x = xa*fb*fc/((fa-fb)(fa-fc)) + xb*fa*fc/((fb-fa)(fb-fc)) + xc*fa*fb/((fc-fa)(fc-fb))
            denom_a = (fa - fb) * (fa - fc)
            denom_b = (fb - fa) * (fb - fc)
            denom_c = (fc - fa) * (fc - fb)
            
            if abs(denom_a) > 1e-30 and abs(denom_b) > 1e-30 and abs(denom_c) > 1e-30:
                x_iqi = (xa * fb * fc / denom_a + 
                         xb * fa * fc / denom_b + 
                         xc * fa * fb / denom_c)
                if not math.isfinite(x_iqi):
                    x_iqi = None
        except (ZeroDivisionError, OverflowError):
            pass
        
        # Safeguard: choose between IQI and secant
        if x_iqi is not None:
            # Measure step sizes relative to the secant step
            step_secant = abs(x_secant - xa)
            step_iqi = abs(x_iqi - xa)
            
            # Use IQI if it's in a reasonable range: not too small (converging) and not too large (diverging)
            if step_secant > 1e-15:
                ratio = step_iqi / step_secant
                # Accept IQI if step is between 0.1x and 2x the secant step
                if 0.1 <= ratio <= 2.0:
                    x_new = x_iqi
                else:
                    x_new = x_secant
            else:
                # Secant step is tiny; trust IQI if it's also close
                if step_iqi < 1e-10 * max(1.0, abs(xa)):
                    x_new = x_iqi
                else:
                    x_new = x_secant
        else:
            x_new = x_secant
        
        # Additional safeguard: if x_new is not finite, use secant
        if not math.isfinite(x_new):
            if abs(fa - fb) > 1e-30:
                x_new = xb - fb * (xb - xa) / (fb - fa)
            else:
                break
        
        # If x_new is extremely close to an existing point, nudge it slightly
        for px, pf in pts:
            if abs(x_new - px) < 1e-14 * max(1.0, abs(x_new)):
                x_new = px + 1e-12 * max(1.0, abs(px))
                break
        
        f_new = safe_f(x_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
        
        # Update points: add new point, keep 3 best by |f|
        pts.append((x_new, f_new))
        pts.sort(key=lambda p: abs(p[1]))
        
        # Remove duplicates (same x)
        kept = []
        for px, pf in pts:
            if kept and abs(px - kept[-1][0]) < 1e-14 * max(1.0, abs(px)):
                if abs(pf) < abs(kept[-1][1]):
                    kept[-1] = (px, pf)
            else:
                kept.append((px, pf))
        
        if len(kept) >= 3:
            pts = kept[:3]
        else:
            break
    
    # Final polish: if we have a good candidate, try a few more secant steps with tighter tolerance
    if abs(best_fval) < 1e-6 and eval_count < max_evals - 10:
        x_cur = best_x
        f_cur = best_fval
        for _ in range(10):
            if time.time() > deadline or eval_count >= max_evals - 5:
                break
            h_polish = 1e-8 * max(1.0, abs(x_cur))
            x_next = x_cur - f_cur * h_polish / (abs(f_cur) + 1e-30)
            if not math.isfinite(x_next):
                break
            f_next = safe_f(x_next)
            if abs(f_next) <= 1e-8:
                return x_next
            if abs(f_next) < abs(f_cur):
                x_cur, f_cur = x_next, f_next
            else:
                break
    
    return best_x
# EVOLVE-BLOCK-END
