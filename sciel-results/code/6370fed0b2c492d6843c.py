import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Global scan + Illinois method root finder.
    
    Strategy:
    1. Perform a uniform grid scan over a large domain centered at x0 to find a sign change.
       This avoids the fragility of frequency estimation.
    2. Once a bracket [a, b] is found, use the Illinois method (a variant of Regula Falsi
       with damping) to converge to the root. This is robust for oscillatory functions
       and does not require derivatives.
    3. If no sign change is found, return the point with the minimum |f(x)| found.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin
    
    x0 = float(x0)
    
    # Phase 1: Global Scan
    # We scan over a range of [-1000, 1000] relative to x0.
    # 2000 points means step size 1.0.
    # If the function is highly oscillatory, we might miss a narrow root, 
    # but "oscillatory" usually implies periods of O(1) to O(10) in these tasks.
    # We adapt the scan range based on x0 magnitude? No, keep it simple and robust.
    # Let's scan from x0 - 1000 to x0 + 1000.
    
    scan_range = 1000.0
    n_points = 2000
    step = 2 * scan_range / n_points
    
    a = x0 - scan_range
    b = x0 + scan_range
    
    prev_x = a
    try:
        prev_f = f(prev_x)
    except:
        prev_f = float('nan')
    
    best_x = prev_x
    best_f = float('inf')
    if math.isfinite(prev_f):
        best_f = abs(prev_f)
        best_x = prev_x
    
    bracket_found = False
    bracket_a = None
    bracket_b = None
    fa = None
    fb = None
    
    for i in range(1, n_points + 1):
        if time.time() > deadline - 5.0:
            break
            
        curr_x = a + i * step
        try:
            curr_f = f(curr_x)
        except:
            curr_f = float('nan')
        
        if not math.isfinite(curr_f):
            continue
        
        if abs(curr_f) < best_f:
            best_f = abs(curr_f)
            best_x = curr_x
            
        if abs(curr_f) <= 1e-8:
            return curr_x
            
        if math.isfinite(prev_f) and prev_f * curr_f < 0:
            bracket_found = True
            bracket_a = prev_x
            bracket_b = curr_x
            fa = prev_f
            fb = curr_f
            break
            
        prev_x = curr_x
        prev_f = curr_f
        
    if not bracket_found:
        # If we didn't find a bracket, try a finer scan around the best point found
        # Or just return best_x. But let's try a local scan around best_x if it's not good enough.
        if best_f > 1e-8:
             # Local scan around best_x
             local_range = max(step, 1.0)
             local_n = 100
             local_step = 2 * local_range / local_n
             local_a = best_x - local_range
             local_b = best_x + local_range
             
             lp_x = local_a
             try:
                 lp_f = f(lp_x)
             except:
                 lp_f = float('nan')
                 
             for j in range(1, local_n + 1):
                 if time.time() > deadline - 2.0:
                     break
                 lc_x = local_a + j * local_step
                 try:
                     lc_f = f(lc_x)
                 except:
                     lc_f = float('nan')
                     
                 if not math.isfinite(lc_f):
                     continue
                     
                 if abs(lc_f) < best_f:
                     best_f = abs(lc_f)
                     best_x = lc_x
                     
                 if abs(lc_f) <= 1e-8:
                     return lc_x
                     
                 if math.isfinite(lp_f) and lp_f * lc_f < 0:
                     bracket_found = True
                     bracket_a = lp_x
                     bracket_b = lc_x
                     fa = lp_f
                     fb = lc_f
                     break
                     
                 lp_x = lc_x
                 lp_f = lc_f
    
    if not bracket_found:
        # No bracket found. Return the best point we have.
        # We can't do much more without risking divergence.
        return best_x
    
    # Phase 2: Illinois Method
    # We have [bracket_a, bracket_b] with f(bracket_a)*f(bracket_b) < 0
    x_low = bracket_a
    x_high = bracket_b
    f_low = fa
    f_high = fb
    
    # Ensure f_low and f_high have opposite signs
    # Illinois method keeps the bracket and replaces the endpoint with the smaller |f|
    # It also damps the function value of the retained endpoint to accelerate convergence.
    
    for _ in range(100):
        if time.time() > deadline - 1.0:
            break
            
        if abs(f_low) <= 1e-8:
            return x_low
        if abs(f_high) <= 1e-8:
            return x_high
            
        # Regula Falsi step
        if f_high - f_low == 0:
            break
        x_new = (x_low * f_high - x_high * f_low) / (f_high - f_low)
        
        # Check if x_new is valid
        if not math.isfinite(x_new):
            break
            
        try:
            f_new = f(x_new)
        except:
            f_new = float('nan')
            
        if not math.isfinite(f_new):
            break
            
        if abs(f_new) <= 1e-8:
            return x_new
            
        if f_new * f_low < 0:
            # Root is in [x_low, x_new]
            x_high = x_new
            f_high = f_new
            # Damp f_low
            f_low *= 0.5
        else:
            # Root is in [x_new, x_high]
            x_low = x_new
            f_low = f_new
            # Damp f_high
            f_high *= 0.5
            
        # Check for convergence of interval
        if abs(x_high - x_low) < 1e-15:
            break
            
    # Return the better of the two endpoints
    if abs(f_low) < abs(f_high):
        return x_low
    else:
        return x_high

# EVOLVE-BLOCK-END
