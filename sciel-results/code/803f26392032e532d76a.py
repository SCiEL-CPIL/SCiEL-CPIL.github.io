"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Derivative-free root finder using secant method with safeguarding,
    targeting smooth polynomial equations. Optimized for minimal function calls.
    """
    deadline = time.time() + 115.0  # safety margin before 120s limit
    x = float(x0)
    
    # Initial evaluation
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Bootstrap: get a second point with a smaller step to reduce overshoot risk
    h = 1e-6 * max(1.0, abs(x))
    x1 = x + h
    f1 = f(x1)
    
    if abs(f1) <= 1e-8:
        return x1
    
    best_x = x if abs(fx) < abs(f1) else x1
    best_fx = min(abs(fx), abs(f1))
    
    # Main secant iteration loop
    for it in range(40):
        if time.time() > deadline:
            break
        
        # Secant step: x_new = x1 - f1 * (x1 - x) / (f1 - fx)
        if f1 == fx:
            # Degenerate case, try bisection-like step
            x_new = (x + x1) / 2.0
        else:
            x_new = x1 - f1 * (x1 - x) / (f1 - fx)
        
        if not math.isfinite(x_new):
            # Fallback: try a small perturbation
            x_new = x1 + 1e-8 * max(1.0, abs(x1))
        
        f_new = f(x_new)
        if abs(f_new) <= 1e-8:
            return x_new
        if abs(f_new) < best_fx:
            best_fx = abs(f_new)
            best_x = x_new
        
        # Update: keep the two most recent points
        x, fx = x1, f1
        x1, f1 = x_new, f_new
        
        # Convergence check on |f| with tighter threshold for early exit
        if abs(f_new) <= 1e-10:
            return x_new
    
    # Final high-precision polish: if we have a good candidate, try a few more
    # secant steps from the best point to push below 1e-8
    if best_fx > 1e-8 and best_fx < 1e-5:
        # Use the two most recent points for continued secant iterations
        xp, fp = x, fx
        xq, fq = x1, f1
        for _ in range(15):
            if time.time() > deadline:
                break
            if abs(fp) <= 1e-8:
                return xp
            if abs(fq) <= 1e-8:
                return xq
            
            # Secant step using the two best points
            if fq == fp:
                break
            x_new = xq - fq * (xq - xp) / (fq - fp)
            
            if not math.isfinite(x_new):
                break
            
            f_new = f(x_new)
            if abs(f_new) <= 1e-8:
                return x_new
            if abs(f_new) < best_fx:
                best_fx = abs(f_new)
                best_x = x_new
            
            # Update: keep the two most recent points
            xp, fp = xq, fq
            xq, fq = x_new, f_new
    
    return best_x
# EVOLVE-BLOCK-END
