"""Root finding via derivative-free minimization of |f(x)|^2 using Nelder-Mead."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses Nelder-Mead simplex method to minimize g(x) = f(x)^2.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    max_evals = 480  # Safety margin for 500 limit
    
    eval_count = 0
    
    def g(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return float('inf')
        eval_count += 1
        fx = f(x)
        if math.isinf(fx) or math.isnan(fx):
            return float('inf')
        return fx * fx

    x0 = float(x0)
    
    # Initial simplex points
    # If x0 is 0, use 1.0 as step, otherwise use 1e-4 * |x0| or 1e-4
    step = 1e-4 * max(1.0, abs(x0))
    
    # Simplex vertices
    x1 = x0
    x2 = x0 + step
    
    f1 = g(x1)
    f2 = g(x2)
    
    best_x = x1 if f1 < f2 else x2
    best_f = min(f1, f2)
    
    # Nelder-Mead parameters
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    
    # Main loop
    for _ in range(200):  # Max iterations
        if time.time() > deadline:
            break
        if eval_count >= max_evals:
            break
        if best_f <= 1e-16:  # Corresponds to |f(x)| <= 1e-8
            break
            
        # Order points: f1 <= f2
        if f1 <= f2:
            x_worst = x2
            f_worst = f2
            x_best = x1
            f_best = f1
            x_mid = x1
            f_mid = f1
        else:
            x_worst = x1
            f_worst = f1
            x_best = x2
            f_best = f2
            x_mid = x2
            f_mid = f2
            
        # Update global best
        if f_best < best_f:
            best_f = f_best
            best_x = x_best
            
        # Centroid of all points except worst
        x_c = x_best
        
        # Reflection
        x_r = x_c + alpha * (x_c - x_worst)
        f_r = g(x_r)
        
        if f_r < f_best:
            # Expansion
            x_e = x_c + gamma * (x_r - x_c)
            f_e = g(x_e)
            if f_e < f_r:
                x_worst, f_worst = x_e, f_e
            else:
                x_worst, f_worst = x_r, f_r
        else:
            if f_r < f_worst:
                # Accept reflection
                x_worst, f_worst = x_r, f_r
            else:
                # Contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c_new = x_c + 0.5 * (x_r - x_c)
                else:
                    # Inside contraction
                    x_c_new = x_c - 0.5 * (x_worst - x_c)
                
                x_cc = x_c_new
                f_cc = g(x_cc)
                
                if f_cc < f_worst:
                    x_worst, f_worst = x_cc, f_cc
                else:
                    # Shrink
                    x_shrink = x_best + rho * (x_worst - x_best)
                    f_shrink = g(x_shrink)
                    x_worst, f_worst = x_shrink, f_shrink
                    
                    # If simplex is too small, try a larger step? 
                    # For now, just break if no progress
                    # Check if simplex collapsed
                    if abs(x_worst - x_best) < 1e-12 * max(1.0, abs(x_best)):
                         break
        
        # Re-evaluate f1 and f2 based on new worst point and the other point
        # We maintain 2 points. Let's say x1 is always the better one? No, let's just swap.
        # It's easier to just keep x1, x2 and re-sort.
        
        # Actually, let's just keep the two points and re-sort them at the start of the next loop.
        # The variables x1, x2, f1, f2 need to be updated.
        # The "other" point is the one that wasn't worst.
        # If x_worst was x1, then the other is x2.
        # If x_worst was x2, then the other is x1.
        
        # Let's just store the current two points in a list and sort
        pts = [(x1, f1), (x2, f2)]
        # Remove the old worst point? No, we replaced it.
        # This logic is getting messy with local vars. Let's use a list of points.
        
        # Let's restart the loop logic with a list of points for clarity.
        # But we already consumed evals. Let's just break and restart the structure?
        # No, let's fix the variable tracking.
        
        # Simpler: Just maintain a list of 2 points.
        # Let's rewrite the loop body to be cleaner.
        
        # Since I can't easily refactor the loop in place without breaking the flow, 
        # let's just break and rely on the initial good points? 
        # No, let's implement the loop correctly using a list.
        
        pass # This pass is just to satisfy the parser, I will replace the loop logic below.

    # The above loop implementation was getting complex. Let's implement a cleaner version.
    # I will replace the entire function body with a cleaner implementation.
    
    return best_x

# Let's rewrite the function properly to avoid the messy in-place updates.

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0
    max_evals = 480
    
    eval_count = 0
    
    def g(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return float('inf')
        eval_count += 1
        try:
            fx = f(x)
            if math.isinf(fx) or math.isnan(fx):
                return float('inf')
            return fx * fx
        except Exception:
            return float('inf')

    x0 = float(x0)
    step = 1e-4 * max(1.0, abs(x0))
    
    # Simplex points
    pts = [
        (x0, g(x0)),
        (x0 + step, g(x0 + step))
    ]
    
    best_x, best_f = min(pts, key=lambda p: p[1])
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    
    for _ in range(200):
        if time.time() > deadline or eval_count >= max_evals:
            break
        if best_f <= 1e-16:
            break
            
        # Sort points
        pts.sort(key=lambda p: p[1])
        x_best, f_best = pts[0]
        x_worst, f_worst = pts[1]
        
        # Update global best
        if f_best < best_f:
            best_f = f_best
            best_x = x_best
            
        # Centroid (only the best point in 1D)
        x_c = x_best
        
        # Reflection
        x_r = x_c + alpha * (x_c - x_worst)
        f_r = g(x_r)
        
        new_worst_pt = None
        
        if f_r < f_best:
            # Expansion
            x_e = x_c + gamma * (x_r - x_c)
            f_e = g(x_e)
            if f_e < f_r:
                new_worst_pt = (x_e, f_e)
            else:
                new_worst_pt = (x_r, f_r)
        else:
            if f_r < f_worst:
                new_worst_pt = (x_r, f_r)
            else:
                # Contraction
                if f_r < f_worst:
                    # Outside contraction (rare in 1D if r is worse than best but better than worst? 
                    # Actually if f_r >= f_worst, it's inside contraction)
                    pass
                # Inside contraction
                x_cc = x_c - 0.5 * (x_worst - x_c)
                f_cc = g(x_cc)
                
                if f_cc < f_worst:
                    new_worst_pt = (x_cc, f_cc)
                else:
                    # Shrink
                    x_shrink = x_best + rho * (x_worst - x_best)
                    f_shrink = g(x_shrink)
                    new_worst_pt = (x_shrink, f_shrink)
                    
                    # Check for collapse
                    if abs(x_shrink - x_best) < 1e-12 * max(1.0, abs(x_best)):
                        break
        
        if new_worst_pt is not None:
            pts[1] = new_worst_pt
            
    return best_x
# EVOLVE-BLOCK-END
