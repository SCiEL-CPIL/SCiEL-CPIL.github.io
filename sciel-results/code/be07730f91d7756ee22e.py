import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder using Nelder-Mead minimization of f(x)^2.
    
    Strategy:
    1. Define a surrogate objective g(x) = f(x)^2.
    2. Initialize a small simplex around x0.
    3. Run Nelder-Mead iterations to minimize g(x).
    4. Return the x with the minimum |f(x)| found.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin for 120s limit
    
    x0 = float(x0)
    f0 = f(x0)
    
    if abs(f0) <= 1e-8:
        return x0
    
    # Objective function to minimize: f(x)^2
    # We track the number of calls and the best result
    best_x = x0
    best_f_sq = f0 * f0
    call_count = 1  # x0 already evaluated
    
    def objective(x):
        nonlocal best_x, best_f_sq, call_count
        fx = f(x)
        call_count += 1
        if not np.isfinite(fx):
            return 1e300
        fx_sq = fx * fx
        if fx_sq < best_f_sq:
            best_f_sq = fx_sq
            best_x = x
        return fx_sq
    
    # Initialize Nelder-Mead Simplex
    # Dimension n=1, so simplex has n+1 = 2 points
    # Start with x0 and x0 + step
    # Choose a step size relative to x0 magnitude or absolute
    scale = max(1.0, abs(x0))
    step = 1e-2 * scale
    
    # Simplex vertices
    # v[0] is the best, v[1] is the worst initially
    v = [x0, x0 + step]
    
    # Evaluate initial simplex
    f_v = [objective(v[0]), objective(v[1])]
    
    # Sort simplex by objective value
    # Since n=1, just ensure v[0] has lower f_v
    if f_v[0] > f_v[1]:
        v[0], v[1] = v[1], v[0]
        f_v[0], f_v[1] = f_v[1], f_v[0]
        
    # Nelder-Mead parameters
    alpha = 1.0   # Reflection
    gamma = 2.0   # Expansion
    rho = 0.5     # Contraction
    sigma = 0.5   # Shrink
    
    max_iter = 200
    
    for i in range(max_iter):
        if time.time() > deadline - 1.0:
            break
            
        # Check convergence on best function value
        if f_v[0] <= (1e-8) ** 2:
            return v[0]
            
        # Check convergence on x difference
        if abs(v[1] - v[0]) < 1e-12 * max(1.0, abs(v[0])):
            break
            
        # 1. Reflection
        # Centroid of all points except the worst (v[1])
        # For n=1, centroid is just v[0]
        centroid = v[0]
        
        # Reflect worst point (v[1]) about centroid
        # x_ref = centroid + alpha * (centroid - x_worst)
        x_ref = centroid + alpha * (centroid - v[1])
        f_ref = objective(x_ref)
        
        if f_ref < f_v[0]:
            # If reflection is better than best, try expansion
            x_exp = centroid + gamma * (x_ref - centroid)
            f_exp = objective(x_exp)
            
            if f_exp < f_ref:
                # Expansion successful
                v[1] = x_exp
                f_v[1] = f_exp
            else:
                # Reflection successful
                v[1] = x_ref
                f_v[1] = f_ref
        elif f_ref < f_v[1]:
            # Reflection is between best and worst
            v[1] = x_ref
            f_v[1] = f_ref
        else:
            # Reflection failed, try contraction
            # Contract towards the best point
            # x_con = centroid + rho * (x_worst - centroid)
            x_con = centroid + rho * (v[1] - centroid)
            f_con = objective(x_con)
            
            if f_con < f_v[1]:
                # Contraction successful
                v[1] = x_con
                f_v[1] = f_con
            else:
                # Shrink
                # Move all points except best towards the best
                for j in range(1, 2):  # Only v[1] needs shrinking
                    v[j] = v[0] + sigma * (v[j] - v[0])
                    f_v[j] = objective(v[j])
                    
        # Re-sort simplex
        if f_v[0] > f_v[1]:
            v[0], v[1] = v[1], v[0]
            f_v[0], f_v[1] = f_v[1], f_v[0]
            
    # After Nelder-Mead, perform a local Secant refinement if we are close enough
    # This leverages the local smoothness found by NM
    if best_f_sq < 1e-4:
        x1 = best_x
        f1 = f(x1)
        call_count += 1
        
        # Choose a small step for secant
        dx = 1e-4 * max(1.0, abs(x1))
        x2 = x1 + dx
        f2 = f(x2)
        call_count += 1
        
        for _ in range(50):
            if time.time() > deadline - 0.5:
                break
            if abs(f1) <= 1e-8:
                return x1
            if abs(f2) <= 1e-8:
                return x2
            if f1 == f2:
                break
                
            x_new = x1 - f1 * (x2 - x1) / (f2 - f1)
            f_new = f(x_new)
            call_count += 1
            
            if abs(f_new) <= 1e-8:
                return x_new
                
            if abs(f_new) < abs(f1):
                x1, f1 = x2, f2
            else:
                x1, f1 = x_new, f_new
            x2, f2 = x_new, f_new
            
            if abs(x_new - x1) < 1e-15:
                break
        
        if abs(f1) < abs(f2):
            return x1
        else:
            return x2

    return best_x
# EVOLVE-BLOCK-END
