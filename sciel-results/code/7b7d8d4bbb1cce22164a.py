"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, using a portfolio of robust methods.
    Uses a deadline guard to stay within the wall-clock budget.
    """
    deadline = time.time() + 145.0
    tol = 1e-10  # target |f(x)| <= 1e-8, use tighter internal tol
    
    x = float(x0)
    best_x = x
    best_f = f(x)
    best_abs = abs(best_f) if best_f is not None and math.isfinite(best_f) else float('inf')
    
    def update_best(x_val, f_val):
        nonlocal best_x, best_f, best_abs
        if x_val is None or f_val is None:
            return
        if not math.isfinite(x_val) or not math.isfinite(f_val):
            return
        abs_f = abs(f_val)
        if abs_f < best_abs:
            best_x = x_val
            best_f = f_val
            best_abs = abs_f
        return abs_f <= tol
    
    # First, check if x0 is already close to a root
    if best_abs <= tol:
        return best_x
    
    # Method 1: Secant method (no derivative needed, good for oscillatory)
    def secant_method(x0, max_iter=200):
        nonlocal best_x, best_f, best_abs
        a = float(x0)
        fa = f(a)
        if fa is None or not math.isfinite(fa):
            return False
        if abs(fa) <= tol:
            update_best(a, fa)
            return True
        update_best(a, fa)
        
        # Choose b with a small offset
        h0 = 1e-4 * max(1.0, abs(a))
        b = a + h0
        fb = f(b)
        if fb is None or not math.isfinite(fb):
            return False
        if abs(fb) <= tol:
            update_best(b, fb)
            return True
        update_best(b, fb)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                return False
            if fa == fb:
                break
            c = b - fb * (b - a) / (fb - fa)
            fc = f(c)
            if fc is None or not math.isfinite(fc):
                break
            if abs(fc) <= tol:
                update_best(c, fc)
                return True
            update_best(c, fc)
            
            # Keep the two points with smallest |f|
            if abs(fc) < abs(fa):
                a, fa = b, fb
                b, fb = c, fc
            else:
                a, fa = b, fb
                b, fb = c, fc
        return best_abs <= tol
    
    # Method 2: Bisection with adaptive bracketing
    def bisection_method(x0, max_iter=200):
        nonlocal best_x, best_f, best_abs
        # Try to find a bracket around x0
        fa = f(x0)
        if fa is None or not math.isfinite(fa):
            return False
        if abs(fa) <= tol:
            update_best(x0, fa)
            return True
        update_best(x0, fa)
        
        # Search for sign change
        step = 1e-2
        found_bracket = False
        for i in range(50):
            if time.time() > deadline:
                return False
            a = x0 - i * step
            b = x0 + i * step
            if a == b:
                break
            fa_val = f(a)
            fb_val = f(b)
            if fa_val is None or not math.isfinite(fa_val):
                continue
            if fb_val is None or not math.isfinite(fb_val):
                continue
            update_best(a, fa_val)
            update_best(b, fb_val)
            if best_abs <= tol:
                return True
            if fa_val * fb_val < 0:
                found_bracket = True
                break
            step *= 2
        
        if not found_bracket:
            return False
        
        # Bisection
        for _ in range(max_iter):
            if time.time() > deadline:
                return False
            c = (a + b) / 2.0
            fc = f(c)
            if fc is None or not math.isfinite(fc):
                break
            if abs(fc) <= tol:
                update_best(c, fc)
                return True
            update_best(c, fc)
            if fa_val * fc < 0:
                b = c
                fb_val = fc
            else:
                a = c
                fa_val = fc
            if abs(b - a) < 1e-15:
                break
        return best_abs <= tol
    
    # Method 3: Newton's method with safe steps
    def newton_method(x0, max_iter=100):
        nonlocal best_x, best_f, best_abs
        x = float(x0)
        fx = f(x)
        if fx is None or not math.isfinite(fx):
            return False
        if abs(fx) <= tol:
            update_best(x, fx)
            return True
        update_best(x, fx)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                return False
            h = 1e-7 * max(1.0, abs(x))
            fxp = f(x + h)
            fxm = f(x - h)
            if fxp is None or not math.isfinite(fxp) or fxm is None or not math.isfinite(fxm):
                break
            d = (fxp - fxm) / (2 * h)
            if d == 0 or not math.isfinite(d):
                break
            step = fx / d
            # Limit step size to prevent divergence
            max_step = 10.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = max_step * (1 if step > 0 else -1)
            x_new = x - step
            if not math.isfinite(x_new):
                break
            fx_new = f(x_new)
            if fx_new is None or not math.isfinite(fx_new):
                break
            if abs(fx_new) <= tol:
                update_best(x_new, fx_new)
                return True
            update_best(x_new, fx_new)
            if abs(fx_new) >= abs(fx):
                # Step made things worse, try smaller step
                x_new = x - step / 2
                fx_new = f(x_new)
                if fx_new is None or not math.isfinite(fx_new):
                    break
                if abs(fx_new) <= tol:
                    update_best(x_new, fx_new)
                    return True
                update_best(x_new, fx_new)
            x, fx = x_new, fx_new
        return best_abs <= tol
    
    # Try methods in order of robustness for oscillatory functions
    if not secant_method(x0):
        if not bisection_method(x0):
            if not newton_method(x0):
                # Last resort: try a few more random-ish perturbations
                for offset in [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1]:
                    if time.time() > deadline:
                        break
                    x_try = x0 + offset
                    fx_try = f(x_try)
                    if fx_try is not None and math.isfinite(fx_try):
                        update_best(x_try, fx_try)
                        if best_abs <= tol:
                            break
    
    # Final check
    if best_abs <= tol:
        return best_x
    
    # Return best found, even if not meeting tol (better than nothing)
    return best_x
# EVOLVE-BLOCK-END
