"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    deadline = time.time() + 140.0  # 140s safety margin
    x = float(x0)
    
    # Try Newton's method first with adaptive step
    best_x = x
    best_fx = f(x)
    best_abs = abs(best_fx)
    
    if best_abs <= 1e-8:
        return x
    
    # Newton's method with bisection fallback
    x0_val = x
    x1_val = x0_val + 1.0
    
    # First, try to find a bracket
    f0 = best_fx
    f1 = f(x1_val)
    calls = 2
    
    # If no sign change, try Newton's method directly
    sign_change = (f0 * f1) <= 0
    
    if not sign_change:
        # Try Newton's method
        for i in range(30):
            if time.time() > deadline:
                break
            fx = f(x)
            calls += 1
            abs_fx = abs(fx)
            if abs_fx < best_abs:
                best_abs = abs_fx
                best_x = x
            if abs_fx <= 1e-8:
                return x
            
            # Central difference for derivative
            h = 1e-5 * max(1.0, abs(x))
            fp = f(x + h)
            fm = f(x - h)
            calls += 2
            d = (fp - fm) / (2 * h)
            
            if d == 0 or not math.isfinite(d):
                break
            
            x_new = x - fx / d
            
            # Damping if step is too large
            if not math.isfinite(x_new):
                break
            
            # Check if we're making progress
            if abs(x_new - x) < 1e-12 * max(1.0, abs(x)):
                break
            
            x = x_new
        
        # If Newton didn't converge well, try to find a bracket by expanding
        if best_abs > 1e-8:
            # Try expanding from x0
            a = x0_val
            b = x0_val
            fa = f(a)
            fb = f(b)
            calls += 2
            
            step = 1.0
            for expand_iter in range(20):
                if time.time() > deadline:
                    break
                
                a = x0_val - step
                b = x0_val + step
                fa = f(a)
                fb = f(b)
                calls += 2
                
                if (fa * fb) <= 0:
                    # Found bracket, use bisection
                    break
                
                step *= 2.0
            else:
                # No bracket found, return best
                return best_x
        
        # Bisection/Secant hybrid
        a, b = x0_val - step, x0_val + step
        fa, fb = f(a), f(b)
        calls += 2
        
        for i in range(50):
            if time.time() > deadline:
                break
            
            c = (a + b) / 2
            fc = f(c)
            calls += 1
            
            abs_fc = abs(fc)
            if abs_fc < best_abs:
                best_abs = abs_fc
                best_x = c
            
            if abs_fc <= 1e-8:
                return c
            
            if (fa * fc) <= 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                break
    
    else:
        # Have a bracket, use Brent-like method
        a, b = x0_val, x1_val
        fa, fb = f0, f1
        
        for i in range(50):
            if time.time() > deadline:
                break
            
            c = (a + b) / 2
            fc = f(c)
            calls += 1
            
            abs_fc = abs(fc)
            if abs_fc < best_abs:
                best_abs = abs_fc
                best_x = c
            
            if abs_fc <= 1e-8:
                return c
            
            if (fa * fc) <= 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                break
    
    return best_x
# EVOLVE-BLOCK-END
