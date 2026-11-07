import time
import math

def find_root(f, x0):
    """
    Finds a root x such that |f(x)| <= 1e-8 using Safeguarded Newton's Method.
    """
    start_time = time.time()
    # The evaluator kills after 150s. We use 100s to be safe.
    deadline = start_time + 100.0
    
    x = float(x0)
    try:
        fx = f(x)
    except Exception:
        return x0 # Or handle error, but task says exception makes it failure. 
                  # We must return a float. If f raises, we can't recover easily.
                  # But f is a smooth polynomial, so it shouldn't raise.
    
    best_x = x
    best_fx = fx
    
    if abs(fx) <= 1e-8:
        return x

    for i in range(200):
        if time.time() > deadline:
            break
            
        # Check for non-finite
        if not math.isfinite(x) or not math.isfinite(fx):
            break
            
        # Adaptive step size for derivative
        # Using a relative step
        h = 1e-6 * max(1.0, abs(x))
        
        # Compute derivative using forward difference
        try:
            fx_h = f(x + h)
        except Exception:
            break
            
        df = (fx_h - fx) / h
        
        # If derivative is effectively zero, we are at a local extremum or flat spot.
        # Try to move in a direction that reduces |f|.
        if abs(df) < 1e-12:
            try:
                fx_minus = f(x - h)
            except Exception:
                break
                
            # Choose the neighbor with smaller |f|
            if abs(fx_minus) < abs(fx):
                x = x - h
                fx = fx_minus
            else:
                x = x + h
                fx = fx_h
                
            if abs(fx) < abs(best_fx):
                best_fx = fx
                best_x = x
            if abs(fx) <= 1e-8:
                return x
            continue
        
        # Newton Step
        step = fx / df
        x_new = x - step
        
        # Safeguard against huge steps
        if not math.isfinite(x_new):
            break
            
        if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
            # Shrink the step
            x_new = x - 0.5 * step
            
        x = x_new
        
        try:
            fx = f(x)
        except Exception:
            break
            
        if abs(fx) < abs(best_fx):
            best_fx = fx
            best_x = x
            
        if abs(fx) <= 1e-8:
            return x
            
    return best_x
