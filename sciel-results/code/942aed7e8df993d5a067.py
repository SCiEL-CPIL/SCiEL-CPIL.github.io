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
    best_x = float(x0)
    best_fx = f(best_x)
    eval_count += 1
    best_abs_fx = abs(best_fx)
    
    if best_abs_fx <= 1e-8:
        return best_x
    
    # Nelder-Mead parameters
    alpha = 0.5  # reflection
    gamma = 2.0  # expansion
    rho = 0.5    # contraction
    sigma = 0.5  # shrink
    
    # Initial simplex: two points (1D problem)
    scale = max(1e-4, abs(float(x0)) * 1e-4)
    x1 = float(x0)
    x2 = x1 + scale
    
    fx1 = f(x1)
    eval_count += 1
    fx2 = f(x2)
    eval_count += 1
    
    # Track best
    if abs(fx1) < best_abs_fx:
        best_abs_fx = abs(fx1)
        best_x = x1
    if abs(fx2) < best_abs_fx:
        best_abs_fx = abs(fx2)
        best_x = x2
    
    if best_abs_fx <= 1e-8:
        return best_x
    
    # Objective: g(x) = f(x)^2
    def g(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return float('inf')
        fx = f(x)
        eval_count += 1
        return fx * fx
    
    # Initialize with already-evaluated points
    g1 = fx1 * fx1
    g2 = fx2 * fx2
    
    # Nelder-Mead iteration
    for _ in range(200):
        if time.time() > deadline:
            break
        if eval_count >= max_evals:
            break
        
        # Sort: worst first
        if g1 <= g2:
            x_best, g_best = x1, g1
            x_worst, g_worst = x2, g2
        else:
            x_best, g_best = x2, g2
            x_worst, g_worst = x1, g1
        
        if g_best <= 1e-16:  # |f| <= 1e-8
            # Update best_x from the one with smaller |f|
            if g1 <= g2:
                best_x = x1
                best_abs_fx = math.sqrt(g1)
            else:
                best_x = x2
                best_abs_fx = math.sqrt(g2)
            return best_x
        
        # Centroid of all but worst (just the best point in 1D)
        centroid = x_best
        
        # Reflection
        x_ref = centroid + alpha * (centroid - x_worst)
        g_ref = g(x_ref)
        
        if g_ref < g_best:
            # Try expansion
            x_exp = centroid + gamma * (centroid - x_worst)
            g_exp = g(x_exp)
            if g_exp < g_ref:
                x_worst, g_worst = x_exp, g_exp
            else:
                x_worst, g_worst = x_ref, g_ref
        elif g_ref < g_worst:
            x_worst, g_worst = x_ref, g_ref
        else:
            # Contraction
            x_con = centroid + rho * (x_worst - centroid)
            g_con = g(x_con)
            if g_con < g_worst:
                x_worst, g_worst = x_con, g_con
            else:
                # Shrink
                x_worst = centroid + sigma * (x_worst - centroid)
                g_worst = g(x_worst)
        
        # Update sorted points
        if g1 <= g2:
            x1, g1 = x_best, g_best
            x2, g2 = x_worst, g_worst
        else:
            x1, g1 = x_worst, g_worst
            x2, g2 = x_best, g_best
        
        # Update global best
        if g1 < g2:
            if abs(math.sqrt(g1)) < best_abs_fx:
                best_abs_fx = abs(math.sqrt(g1))
                best_x = x1
        else:
            if abs(math.sqrt(g2)) < best_abs_fx:
                best_abs_fx = abs(math.sqrt(g2))
                best_x = x2
        
        if best_abs_fx <= 1e-8:
            return best_x
    
    # Final polish: try a few more evaluations around best_x
    if eval_count < max_evals - 10:
        # Try Newton-like step with finite difference if we have enough evals
        for _ in range(3):
            if time.time() > deadline or eval_count >= max_evals - 2:
                break
            fx = f(best_x)
            eval_count += 1
            if abs(fx) <= 1e-8:
                return best_x
            h = 1e-7 * max(1.0, abs(best_x))
            fx_plus = f(best_x + h)
            eval_count += 1
            d = (fx_plus - fx) / h
            if abs(d) > 1e-30:
                new_x = best_x - fx / d
                fx_new = f(new_x)
                eval_count += 1
                if abs(fx_new) < abs(fx):
                    best_x = new_x
                    best_abs_fx = abs(fx_new)
                    if best_abs_fx <= 1e-8:
                        return best_x
                else:
                    break
    
    return best_x
# EVOLVE-BLOCK-END
