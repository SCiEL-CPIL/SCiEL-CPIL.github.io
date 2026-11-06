"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def find_root(f, x0):
    """Newton's method with damping, central differences, and bisection fallback."""
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    x = float(x0)
    
    # Track best result
    best_x = x
    best_fx = f(x)
    best_abs = abs(best_fx)
    
    for it in range(200):
        # Time guard
        if time.time() - start_time > DEADLINE:
            break
        
        fx = f(x)
        abs_fx = abs(fx)
        
        if abs_fx < best_abs:
            best_x = x
            best_fx = fx
            best_abs = abs_fx
        
        if abs_fx <= 1e-10:
            return x
        
        # Central difference derivative
        h = 1e-7 * max(1.0, abs(x))
        fp = f(x + h)
        fm = f(x - h)
        d = (fp - fm) / (2.0 * h)
        
        if d == 0 or not (d == d):  # zero or NaN
            # Try bisection-like step
            # We need to bracket; use a small step in direction that reduces |f|
            step = 1e-4 * max(1.0, abs(x))
            x_trial = x + step
            f_trial = f(x_trial)
            if abs(f_trial) < abs_fx:
                x = x_trial
            else:
                x = x - step
            continue
        
        # Newton step with damping
        dx = fx / d
        x_new = x - dx
        
        # Damp if step is too large
        max_step = 10.0 * max(1.0, abs(x))
        if abs(dx) > max_step:
            dx = max_step * (1.0 if dx > 0 else -1.0)
            x_new = x - dx
        
        # Check if step is valid (finite)
        if not (x_new == x_new) or abs(x_new) > 1e300:
            # Damp more aggressively
            dx = 0.1 * fx / d
            x_new = x - dx
            if not (x_new == x_new) or abs(x_new) > 1e300:
                break
        
        x = x_new
    
    # Final polish: if we have a good bracket, do a few more iterations
    if best_abs <= 1e-8:
        return best_x
    
    # Try a few more Newton iterations from best_x
    x = best_x
    for _ in range(20):
        if time.time() - start_time > DEADLINE:
            break
        fx = f(x)
        if abs(fx) <= 1e-10:
            return x
        h = 1e-8 * max(1.0, abs(x))
        d = (f(x + h) - f(x - h)) / (2.0 * h)
        if d == 0:
            break
        dx = fx / d
        if abs(dx) > 1.0 * max(1.0, abs(x)):
            dx = 1.0 * max(1.0, abs(x)) * (1.0 if dx > 0 else -1.0)
        x = x - dx
        if abs(f(x)) < best_abs:
            best_x = x
            best_abs = abs(f(x))
    
    return best_x
# EVOLVE-BLOCK-END
