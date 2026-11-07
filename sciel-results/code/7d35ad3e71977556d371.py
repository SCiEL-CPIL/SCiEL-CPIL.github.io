"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    x = float(x0)
    best_x = x
    best_fx = f(x)
    best_abs_fx = abs(best_fx)
    
    if best_abs_fx <= 1e-8:
        return x
    
    # Try Newton's method with safeguarding
    for i in range(40):
        if time.time() >= deadline:
            break
        
        fx = best_fx
        if abs(fx) <= 1e-12:
            return x
        
        # Central difference for derivative
        h = 1e-7 * max(1.0, abs(x))
        d = (f(x + h) - f(x - h)) / (2 * h)
        
        if d == 0 or not math.isfinite(d):
            break
        
        new_x = x - fx / d
        
        # Check if new_x is reasonable
        if not math.isfinite(new_x):
            break
        
        # Damped step if needed
        step = new_x - x
        if abs(step) > 10 * max(1.0, abs(x)):
            # Too large a step, damp it
            new_x = x + 0.5 * step
        
        new_fx = f(new_x)
        new_abs = abs(new_fx)
        
        if not math.isfinite(new_fx):
            # If new point is bad, try a smaller step
            new_x = x - 0.5 * fx / d
            new_fx = f(new_x)
            new_abs = abs(new_fx)
            if not math.isfinite(new_fx):
                break
        
        # Accept if it improves, or if we're getting closer to zero
        if new_abs < best_abs_fx or i < 5:
            x = new_x
            best_x = x
            best_fx = new_fx
            best_abs_fx = new_abs
        else:
            # Newton is diverging, break out
            break
    
    if best_abs_fx <= 1e-8:
        return best_x
    
    # If Newton didn't work well, try bisection with adaptive bracketing
    # First, try to find a bracket around x0
    a = x0 - 1.0
    b = x0 + 1.0
    
    fa = f(a)
    fb = f(b)
    
    # Expand bracket if needed
    if math.isfinite(fa) and math.isfinite(fb):
        if fa * fb > 0:
            # No sign change, try expanding
            for expand in range(10):
                if time.time() >= deadline:
                    break
                a = x0 - (2.0 ** (expand + 1))
                b = x0 + (2.0 ** (expand + 1))
                fa = f(a)
                fb = f(b)
                if not (math.isfinite(fa) and math.isfinite(fb)):
                    break
                if fa * fb <= 0:
                    break
    
    # Check if we have a valid bracket
    if math.isfinite(fa) and math.isfinite(fb) and fa * fb <= 0:
        # Bisection
        for i in range(60):
            if time.time() >= deadline:
                break
            
            mid = (a + b) / 2.0
            fm = f(mid)
            
            if not math.isfinite(fm):
                break
            
            abs_fm = abs(fm)
            if abs_fm < best_abs_fx:
                best_x = mid
                best_abs_fx = abs_fm
            
            if abs_fm <= 1e-12:
                return mid
            
            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
            
            if abs(b - a) < 1e-14 * max(1.0, abs(mid)):
                break
    
    # Final polish with Newton if we have a good starting point
    if best_abs_fx > 1e-8 and best_abs_fx < 1e-4:
        x = best_x
        for i in range(20):
            if time.time() >= deadline:
                break
            
            fx = f(x)
            if abs(fx) <= 1e-12:
                return x
            
            h = 1e-8 * max(1.0, abs(x))
            d = (f(x + h) - f(x - h)) / (2 * h)
            
            if d == 0 or not math.isfinite(d):
                break
            
            new_x = x - fx / d
            if not math.isfinite(new_x):
                break
            
            new_fx = f(new_x)
            if not math.isfinite(new_fx):
                break
            
            if abs(new_fx) < abs(fx):
                x = new_x
                best_x = x
                best_abs_fx = abs(new_fx)
            else:
                break
    
    return best_x
# EVOLVE-BLOCK-END
