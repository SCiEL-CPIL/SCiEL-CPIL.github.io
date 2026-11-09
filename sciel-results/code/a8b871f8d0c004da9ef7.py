"""Root finder for oscillatory equations."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    tol = 1e-8
    x = float(x0)
    
    # Deadline guard: 150s wall-clock limit, use 140s safety margin
    t_start = time.time()
    deadline = t_start + 140.0
    
    # Count calls
    call_count = [0]
    
    def call_f(val):
        call_count[0] += 1
        return f(val)
    
    # First evaluate at x0
    fx = call_f(x)
    if not math.isfinite(fx):
        # Try nearby points
        for dx in [1e-6, 1e-4, 1e-2, 0.1, 1.0]:
            fx = call_f(x + dx)
            if math.isfinite(fx):
                x = x + dx
                break
        if not math.isfinite(fx):
            return x
    
    if abs(fx) <= tol:
        return x
    
    # Steffensen's method: quadratic convergence without derivative
    # x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
    # But we need two starting points. Let's use a small perturbation.
    
    # Try Steffensen's with adaptive step
    x_prev = x
    x_curr = x + 1e-8 * max(1.0, abs(x))
    
    # We already have f(x_prev) = fx
    fx_prev = fx
    fx_curr = call_f(x_curr)
    
    if not math.isfinite(fx_curr):
        x_curr = x + 1e-4 * max(1.0, abs(x))
        fx_curr = call_f(x_curr)
    
    if abs(fx_curr) <= tol:
        return x_curr
    
    # Steffensen's iteration
    best_x = x
    best_fx = abs(fx)
    
    for i in range(100):
        if time.time() > deadline - 5.0:
            break
        
        # Check if we're making progress
        if abs(fx_curr) < best_fx:
            best_x = x_curr
            best_fx = abs(fx_curr)
        
        if abs(fx_curr) <= tol:
            return x_curr
        
        # Steffensen step
        denom = fx_curr - fx_prev
        if abs(denom) < 1e-30:
            # Degenerate, try a different perturbation
            x_curr = x_prev + 1e-4 * max(1.0, abs(x_prev))
            fx_curr = call_f(x_curr)
            if not math.isfinite(fx_curr):
                x_curr = x_prev + 1e-2 * max(1.0, abs(x_prev))
                fx_curr = call_f(x_curr)
            if abs(fx_curr) <= tol:
                return x_curr
            if not math.isfinite(fx_curr):
                break
            denom = fx_curr - fx_prev
            if abs(denom) < 1e-30:
                break
        
        x_new = x_curr - fx_curr * (x_curr - x_prev) / denom
        
        if not math.isfinite(x_new):
            # Fallback: bisection-like step
            x_new = 0.5 * (x_prev + x_curr)
        
        x_prev = x_curr
        fx_prev = fx_curr
        x_curr = x_new
        fx_curr = call_f(x_curr)
        
        if not math.isfinite(fx_curr):
            # Try to recover
            x_curr = 0.5 * (x_prev + x_new)
            fx_curr = call_f(x_curr)
            if not math.isfinite(fx_curr):
                break
        
        # Check for convergence
        if abs(x_curr - x_prev) < 1e-15 * max(1.0, abs(x_curr)):
            if abs(fx_curr) <= tol:
                return x_curr
            # Try one more refinement
            break
    
    # If Steffensen didn't converge, try a bracketing approach
    # First, try to find a sign change
    a = x
    fa = fx
    b = None
    fb = None
    
    # Try expanding bracket
    for delta in [1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]:
        if time.time() > deadline - 5.0:
            break
        
        # Try x + delta
        b_try = a + delta
        fb_try = call_f(b_try)
        if math.isfinite(fb_try):
            if abs(fb_try) <= tol:
                return b_try
            if fa * fb_try < 0:
                b = b_try
                fb = fb_try
                break
        
        # Try x - delta
        b_try = a - delta
        fb_try = call_f(b_try)
        if math.isfinite(fb_try):
            if abs(fb_try) <= tol:
                return b_try
            if fa * fb_try < 0:
                b = b_try
                fb = fb_try
                break
    
    # If we found a bracket, use bisection + secant hybrid
    if b is not None and fb is not None:
        # Bisection with secant acceleration
        lo, hi = a, b
        flo, fhi = fa, fb
        # Ensure flo*fhi < 0
        if flo * fhi > 0:
            lo, hi = hi, lo
            flo, fhi = fhi, flo
        
        # Keep track of best
        best_x_bracket = 0.5 * (lo + hi)
        best_fx_bracket = abs(call_f(best_x_bracket))
        
        for i in range(200):
            if time.time() > deadline - 5.0:
                break
            
            mid = 0.5 * (lo + hi)
            fmid = call_f(mid)
            
            if not math.isfinite(fmid):
                break
            
            if abs(fmid) <= tol:
                return mid
            
            if abs(fmid) < best_fx_bracket:
                best_x_bracket = mid
                best_fx_bracket = abs(fmid)
            
            # Update bracket
            if flo * fmid <= 0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid
            
            # Convergence check
            if abs(hi - lo) < 1e-15 * max(1.0, abs(mid)):
                break
            
            # Secant step if possible
            if i >= 1 and abs(fmid - fhi) > 1e-30:
                x_secant = hi - fhi * (hi - mid) / (fhi - fmid)
                if lo < x_secant < hi and math.isfinite(x_secant):
                    f_secant = call_f(x_secant)
                    if math.isfinite(f_secant):
                        if abs(f_secant) <= tol:
                            return x_secant
                        if abs(f_secant) < best_fx_bracket:
                            best_x_bracket = x_secant
                            best_fx_bracket = abs(f_secant)
                        if flo * f_secant <= 0:
                            hi = x_secant
                            fhi = f_secant
                        else:
                            lo = x_secant
                            flo = f_secant
        
        if best_fx_bracket <= tol:
            return best_x_bracket
        if best_fx_bracket < best_fx:
            best_x = best_x_bracket
            best_fx = best_fx_bracket
    
    # Final check: if best_fx is close, do a high-precision polish
    if best_fx <= 1e-4:
        # Use Newton with finite difference for polish
        x = best_x
        for i in range(30):
            if time.time() > deadline - 2.0:
                break
            fx = call_f(x)
            if not math.isfinite(fx):
                break
            if abs(fx) <= tol:
                return x
            h = 1e-8 * max(1.0, abs(x))
            d = (call_f(x + h) - fx) / h
            if abs(d) < 1e-30:
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            x = x_new
    
    return best_x
# EVOLVE-BLOCK-END
