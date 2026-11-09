"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    # Deadline guard: kill after ~115s to stay under 120s wall-clock
    t_start = time.time()
    deadline = 115.0
    
    eval_count = 0
    max_evals = 500
    
    def f_counted(x):
        nonlocal eval_count
        eval_count += 1
        if eval_count > max_evals:
            raise ValueError("Max evaluations exceeded")
        return f(x)
    
    x = float(x0)
    
    # First evaluation
    fx = f_counted(x)
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)
    
    if best_abs_fx <= 1e-8:
        return x
    
    # IQI (Ill-conditioned Quasi-Newton / Secant-like) method
    # We maintain two points and use secant-like updates, but with careful
    # handling for ill-conditioning.
    
    # Start with a small perturbation to get a second point
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    f1 = f_counted(x1)
    
    if abs(f1) <= 1e-8:
        return x1
    
    # Track best
    if abs(f1) < best_abs_fx:
        best_x = x1
        best_abs_fx = abs(f1)
    
    # Secant/IQI iteration
    # Keep (x0, f0) and (x1, f1)
    x0_s = x
    f0_s = fx
    x1_s = x1
    f1_s = f1
    
    for i in range(max_evals - 2):
        # Check time
        if time.time() - t_start > deadline:
            break
        
        # Check if we've converged
        if abs(f1_s) <= 1e-8:
            return x1_s
        if abs(f0_s) <= 1e-8:
            return x0_s
        
        # Compute secant slope
        df = f1_s - f0_s
        dx = x1_s - x0_s
        
        if abs(df) < 1e-15:
            # Nearly flat; try a different perturbation
            break
        
        if abs(dx) < 1e-15:
            # Points too close; perturb
            break
        
        slope = df / dx
        
        # Next point via secant
        x_new = x1_s - f1_s / slope
        
        # Check for NaN/Inf
        if not math.isfinite(x_new):
            break
        
        # Evaluate
        f_new = f_counted(x_new)
        
        if not math.isfinite(f_new):
            break
        
        # Update best
        if abs(f_new) < best_abs_fx:
            best_x = x_new
            best_abs_fx = abs(f_new)
            if best_abs_fx <= 1e-8:
                return x_new
        
        # Update the pair: keep the two most recent points
        x0_s = x1_s
        f0_s = f1_s
        x1_s = x_new
        f1_s = f_new
        
        # Check convergence
        if abs(f1_s) <= 1e-8:
            return x1_s
        if abs(f0_s) <= 1e-8:
            return x0_s
    
    # If secant didn't work well, try a local search / golden section on |f(x)|
    # to find a minimum of |f| near the current best point
    if time.time() - t_start < deadline - 5.0:
        # Golden section search on |f(x)| in a window around best_x
        a = best_x - 1e-3 * max(1.0, abs(best_x))
        b = best_x + 1e-3 * max(1.0, abs(best_x))
        
        # Ensure we have valid evaluations
        try:
            fa = f_counted(a)
            fb = f_counted(b)
        except ValueError:
            return best_x
        
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
        
        if abs(fa) < best_abs_fx:
            best_x = a
            best_abs_fx = abs(fa)
        if abs(fb) < best_abs_fx:
            best_x = b
            best_abs_fx = abs(fb)
        
        # Golden section
        gr = (math.sqrt(5) - 1) / 2  # ~0.618
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        try:
            fc = f_counted(c)
            fd = f_counted(d)
        except ValueError:
            return best_x
        
        if abs(fc) < best_abs_fx:
            best_x = c
            best_abs_fx = abs(fc)
        if abs(fd) < best_abs_fx:
            best_x = d
            best_abs_fx = abs(fd)
        
        # Iterate a few times
        for _ in range(20):
            if time.time() - t_start > deadline - 2.0:
                break
            if eval_count >= max_evals - 5:
                break
            
            if abs(fc) < abs(fd):
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                try:
                    fc = f_counted(c)
                except ValueError:
                    break
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                try:
                    fd = f_counted(d)
                except ValueError:
                    break
            
            if abs(fc) < best_abs_fx:
                best_x = c
                best_abs_fx = abs(fc)
            if abs(fd) < best_abs_fx:
                best_x = d
                best_abs_fx = abs(fd)
            
            if best_abs_fx <= 1e-8:
                return best_x
    
    return best_x
# EVOLVE-BLOCK-END
