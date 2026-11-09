"""Root finder for 3 fixed smooth polynomial equations."""
import time
import math

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Strategy:
      1. Try Newton's method with finite-difference derivative (quadratic convergence).
      2. If Newton fails or diverges, fall back to a robust hybrid:
         - Use a secant/Brent-like approach with sign changes.
         - Expand the bracket if needed.
    Keep call count low. Return best x found.
    """
    deadline = time.time() + 110.0  # safety margin before 120s kill
    best_x = float(x0)
    best_abs_fx = float('inf')
    
    def update_best(x, fx):
        nonlocal best_x, best_abs_fx
        if math.isfinite(x) and math.isfinite(fx):
            afx = abs(fx)
            if afx < best_abs_fx:
                best_abs_fx = afx
                best_x = x
    
    # --- Phase 1: Newton with finite-difference derivative ---
    x = float(x0)
    newton_success = False
    
    for i in range(40):
        if time.time() > deadline - 10.0:
            break
        
        fx = f(x)
        if not math.isfinite(fx):
            break
        update_best(x, fx)
        
        if abs(fx) <= 1e-12:
            newton_success = True
            return x
        
        h = 1e-7 * max(1.0, abs(x))
        fx_plus = f(x + h)
        if not math.isfinite(fx_plus):
            break
        update_best(x + h, fx_plus)
        
        d = (fx_plus - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        
        x_new = x - fx / d
        
        # Damping if step is too large
        if abs(x_new - x) > 100 * (1.0 + abs(x)):
            # Try a smaller step
            x_new = x - 0.5 * fx / d
            if abs(x_new - x) > 50 * (1.0 + abs(x)):
                x_new = x - 0.1 * fx / d
        
        if not math.isfinite(x_new):
            break
        
        x = x_new
    
    if newton_success:
        return best_x
    
    # --- Phase 2: Secant method (uses 2 function calls per iteration after initial) ---
    # Use the last two points from Newton if available, else start fresh
    # We'll do a simple secant with fallback to bisection-like bracketing
    
    # Get two initial points
    x1 = float(x0)
    x2 = x1 + 0.1 * (1.0 + abs(x1))
    
    f1 = f(x1)
    f2 = f(x2)
    update_best(x1, f1)
    update_best(x2, f2)
    
    if abs(f1) <= 1e-8:
        return x1
    if abs(f2) <= 1e-8:
        return x2
    
    # If we have a sign change, use bisection + secant hybrid
    # Otherwise, try secant anyway
    for i in range(50):
        if time.time() > deadline - 5.0:
            break
        
        if not math.isfinite(f1) or not math.isfinite(f2):
            break
        
        # Secant step
        if f2 != f1:
            x_sec = x1 - f1 * (x2 - x1) / (f2 - f1)
        else:
            x_sec = (x1 + x2) / 2.0
        
        if not math.isfinite(x_sec):
            break
        
        f_sec = f(x_sec)
        update_best(x_sec, f_sec)
        
        if abs(f_sec) <= 1e-12:
            return x_sec
        
        # Update points: keep the one with smaller |f|
        if abs(f_sec) < abs(f1):
            x1, f1 = x2, f2
            x2, f2 = x_sec, f_sec
        elif abs(f_sec) < abs(f2):
            x2, f2 = x_sec, f_sec
        else:
            # Keep x1, x2, but try expanding
            # If both f1 and f2 have same sign and we're not converging, expand
            if f1 * f2 > 0:
                # No sign change, expand the interval
                if abs(x2 - x1) < 1e-10:
                    break
                # Expand in direction away from the point with smaller |f|
                if abs(f1) < abs(f2):
                    x2 = x2 + 0.5 * (x2 - x1)
                    f2 = f(x2)
                    update_best(x2, f2)
                else:
                    x1 = x1 - 0.5 * (x2 - x1)
                    f1 = f(x1)
                    update_best(x1, f1)
            else:
                # Sign change: use bisection
                x_mid = (x1 + x2) / 2.0
                f_mid = f(x_mid)
                update_best(x_mid, f_mid)
                if abs(f_mid) <= 1e-12:
                    return x_mid
                if f1 * f_mid <= 0:
                    x2, f2 = x_mid, f_mid
                else:
                    x1, f1 = x_mid, f_mid
    
    return best_x

# EVOLVE-BLOCK-END
