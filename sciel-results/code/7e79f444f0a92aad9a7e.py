"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, using a portfolio of robust methods.
    Uses a deadline guard to stay within the wall-clock budget.
    """
    deadline = time.time() + 115.0  # safety margin before 120s limit
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
    
    # Check if x0 is already close to a root
    if best_abs <= tol:
        return best_x
    
    # Method 1: Coarse-to-fine bracket search, then bisection/secant
    def bracket_and_solve(x0):
        nonlocal best_x, best_f, best_abs
        if time.time() > deadline:
            return best_abs <= tol
        
        # Coarse-to-fine: try larger steps first to find a bracket quickly
        base_h = 1e-3 * max(1.0, abs(x0))
        fa = f(x0)
        if fa is None or not math.isfinite(fa):
            return best_abs <= tol
        update_best(x0, fa)
        if abs(fa) <= tol:
            return True
        
        # Try expanding brackets with coarse steps first
        scales = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        bracket_found = False
        a = b = None
        fa_val = fb_val = None
        
        for scale in scales:
            if time.time() > deadline:
                return best_abs <= tol
            h = base_h * scale
            a = x0 - h
            b = x0 + h
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
                bracket_found = True
                break
        
        if not bracket_found:
            return best_abs <= tol
        
        # Now do bisection within the bracket
        # First, try secant for faster convergence
        if time.time() > deadline:
            return best_abs <= tol
        
        # Use secant method within the bracket
        x1, f1 = a, fa_val
        x2, f2 = b, fb_val
        
        for _ in range(80):
            if time.time() > deadline:
                return best_abs <= tol
            if abs(f2) <= tol:
                update_best(x2, f2)
                return True
            if abs(f1) <= tol:
                update_best(x1, f1)
                return True
            denom = f2 - f1
            if denom == 0:
                break
            x3 = x1 - f1 * (x2 - x1) / denom
            # Keep within bracket if possible
            if x3 < a or x3 > b:
                # Fall back to bisection
                x3 = (a + b) / 2.0
            f3 = f(x3)
            if f3 is None or not math.isfinite(f3):
                break
            update_best(x3, f3)
            if abs(f3) <= tol:
                return True
            
            # Update bracket to maintain sign change
            if f1 * f3 < 0:
                x2, f2 = x3, f3
                b = x3
            else:
                x1, f1 = x3, f3
                a = x3
        
        return best_abs <= tol
    
    # Method 2: Secant method (no bracket needed)
    def secant_method(x0):
        nonlocal best_x, best_f, best_abs
        if time.time() > deadline:
            return best_abs <= tol
        
        x1 = float(x0)
        f1 = f(x1)
        if f1 is None or not math.isfinite(f1):
            return best_abs <= tol
        update_best(x1, f1)
        if abs(f1) <= tol:
            return True
        
        h0 = 1e-4 * max(1.0, abs(x1))
        x2 = x1 + h0
        f2 = f(x2)
        if f2 is None or not math.isfinite(f2):
            return best_abs <= tol
        update_best(x2, f2)
        if abs(f2) <= tol:
            return True
        
        for _ in range(150):
            if time.time() > deadline:
                return best_abs <= tol
            if abs(f2) <= tol:
                update_best(x2, f2)
                return True
            if abs(f1) <= tol:
                update_best(x1, f1)
                return True
            denom = f2 - f1
            if denom == 0:
                break
            x3 = x1 - f1 * (x2 - x1) / denom
            f3 = f(x3)
            if f3 is None or not math.isfinite(f3):
                break
            update_best(x3, f3)
            if abs(f3) <= tol:
                return True
            if not math.isfinite(x3):
                break
            x1, f1 = x2, f2
            x2, f2 = x3, f3
        
        return best_abs <= tol
    
    # Method 3: Newton's method with finite-difference derivative
    def newton_method(x0):
        nonlocal best_x, best_f, best_abs
        if time.time() > deadline:
            return best_abs <= tol
        
        x = float(x0)
        fx = f(x)
        if fx is None or not math.isfinite(fx):
            return best_abs <= tol
        update_best(x, fx)
        if abs(fx) <= tol:
            return True
        
        for _ in range(80):
            if time.time() > deadline:
                return best_abs <= tol
            h = 1e-7 * max(1.0, abs(x))
            fxp = f(x + h)
            fxm = f(x - h)
            if fxp is None or not math.isfinite(fxp) or fxm is None or not math.isfinite(fxm):
                break
            d = (fxp - fxm) / (2 * h)
            if d == 0 or not math.isfinite(d):
                break
            step = fx / d
            max_step = 10.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = max_step * (1 if step > 0 else -1)
            x_new = x - step
            if not math.isfinite(x_new):
                break
            fx_new = f(x_new)
            if fx_new is None or not math.isfinite(fx_new):
                break
            update_best(x_new, fx_new)
            if abs(fx_new) <= tol:
                return True
            if abs(fx_new) >= abs(fx):
                # Step made things worse, try smaller step
                x_new = x - step / 2
                fx_new = f(x_new)
                if fx_new is None or not math.isfinite(fx_new):
                    break
                update_best(x_new, fx_new)
                if abs(fx_new) <= tol:
                    return True
            x, fx = x_new, fx_new
        
        return best_abs <= tol
    
    # Final polish: if we have a good candidate, do high-precision bisection
    def polish(x_candidate):
        nonlocal best_x, best_f, best_abs
        if time.time() > deadline:
            return
        fc = f(x_candidate)
        if fc is None or not math.isfinite(fc):
            return
        update_best(x_candidate, fc)
        if abs(fc) <= tol:
            return
        
        # Try to tighten with local secant
        h = 1e-6 * max(1.0, abs(x_candidate))
        x1 = x_candidate - h
        x2 = x_candidate + h
        f1 = f(x1)
        f2 = f(x2)
        if f1 is None or not math.isfinite(f1) or f2 is None or not math.isfinite(f2):
            return
        update_best(x1, f1)
        update_best(x2, f2)
        
        if f1 * f2 < 0:
            # Bisection polish
            a, b = x1, x2
            fa, fb = f1, f2
            for _ in range(50):
                if time.time() > deadline:
                    return
                m = (a + b) / 2.0
                fm = f(m)
                if fm is None or not math.isfinite(fm):
                    break
                update_best(m, fm)
                if abs(fm) <= tol:
                    return
                if fa * fm < 0:
                    b = m
                    fb = fm
                else:
                    a = m
                    fa = fm
                if abs(b - a) < 1e-15:
                    break
    
    # Run strategies in order of expected efficiency
    # 1. Try bracket search first (most reliable for oscillatory)
    if not bracket_and_solve(x0):
        # 2. Try secant
        if not secant_method(x0):
            # 3. Try Newton
            if not newton_method(x0):
                # 4. Try a few perturbations
                for delta in [1e-3, -1e-3, 1e-2, -1e-2, 1e-1, -1e-1, 1.0, -1.0]:
                    if time.time() > deadline:
                        break
                    x_try = x0 + delta * max(1.0, abs(x0))
                    f_try = f(x_try)
                    if f_try is not None and math.isfinite(f_try):
                        update_best(x_try, f_try)
                        if best_abs <= tol:
                            break
    
    # Final polish on best candidate
    if best_abs > tol and best_abs < 1.0:
        polish(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
