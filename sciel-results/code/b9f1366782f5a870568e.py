"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    x = float(x0)
    # Use a tighter tolerance to save calls, but ensure we meet the 1e-8 requirement.
    # 1e-8 is the target.
    for _ in range(50):
        if not math.isfinite(x):
            break
        try:
            fx = f(x)
        except:
            break
        if not math.isfinite(fx):
            break
        if abs(fx) <= 1e-8:
            return x
        
        # Central difference for better accuracy
        h = 1e-5 * max(1.0, abs(x))
        try:
            f_xp = f(x + h)
            f_xm = f(x - h)
        except:
            break
        if not math.isfinite(f_xp) or not math.isfinite(f_xm):
            break
            
        d = (f_xp - f_xm) / (2.0 * h)
        
        if abs(d) < 1e-12:
            # Derivative too small, Newton step will be huge/unstable
            break
            
        x_new = x - fx / d
        
        if not math.isfinite(x_new):
            break
            
        # Damping: prevent large jumps that might diverge
        # If step is larger than 1.0, limit it to 1.0 in the direction of the step
        step = x_new - x
        if abs(step) > 1.0:
            x_new = x + (1.0 if step > 0 else -1.0)
            
        x = x_new
        
    return x
# EVOLVE-BLOCK-END
