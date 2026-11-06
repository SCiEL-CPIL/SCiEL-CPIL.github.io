"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find root of f near x0 using Nelder-Mead minimization of |f(x)|.
    Returns x with |f(x)| <= 1e-8 or best effort within 500 evals.
    """
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    eval_count = 0
    MAX_EVALS = 500
    
    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        result = f(x)
        eval_count += 1
        return result
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Objective function: minimize |f(x)|
    def objective(x):
        try:
            val = call_f(x)
            if not math.isfinite(val):
                return float('inf')
            return abs(val)
        except:
            return float('inf')
    
    # Nelder-Mead Algorithm
    def nelder_mead(x0, max_iter=200):
        # Initial simplex: x0, x0 + h, x0 - h
        h = 1e-4 * max(1.0, abs(x0))
        points = [
            [float(x0)],
            [float(x0) + h],
            [float(x0) - h]
        ]
        
        # Evaluate initial points
        values = [objective(p[0]) for p in points]
        
        # Sort points by value
        order = sorted(range(len(points)), key=lambda i: values[i])
        points = [points[i] for i in order]
        values = [values[i] for i in order]
        
        best_x = points[0][0]
        best_val = values[0]
        
        if best_val <= 1e-8:
            return best_x, best_val
        
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrinkage
        
        for _ in range(max_iter):
            if time_left() < 1.0:
                break
            
            # Check convergence
            if values[0] <= 1e-8:
                return points[0][0], values[0]
            
            # Check if simplex is small enough
            spread = abs(points[0][0] - points[1][0])
            if spread < 1e-12 * max(1.0, abs(points[0][0])):
                break
            
            # Centroid of all points except the worst
            centroid = (points[0][0] + points[1][0]) / 2.0
            
            # 1. Reflection
            x_r = centroid + alpha * (centroid - points[2][0])
            f_r = objective(x_r)
            
            if f_r < values[0]:
                # 2. Expansion
                x_e = centroid + gamma * (x_r - centroid)
                f_e = objective(x_e)
                if f_e < f_r:
                    points[2] = [x_e]
                    values[2] = f_e
                else:
                    points[2] = [x_r]
                    values[2] = f_r
            elif f_r < values[1]:
                points[2] = [x_r]
                values[2] = f_r
            else:
                # 3. Contraction
                if f_r < values[2]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (x_r - centroid)
                
                f_c = objective(x_c)
                
                if f_c < min(f_r, values[2]):
                    points[2] = [x_c]
                    values[2] = f_c
                else:
                    # 4. Shrinkage
                    for i in range(1, len(points)):
                        points[i][0] = points[0][0] + sigma * (points[i][0] - points[0][0])
                        values[i] = objective(points[i][0])
                    
                    # Re-sort
                    order = sorted(range(len(points)), key=lambda i: values[i])
                    points = [points[i] for i in order]
                    values = [values[i] for i in order]
            
            # Update best
            if values[0] < best_val:
                best_val = values[0]
                best_x = points[0][0]
            
            # Re-sort after each iteration (except shrinkage which already sorted)
            if not (f_c < min(f_r, values[2]) if 'f_c' in locals() and 'f_r' in locals() else False):
                order = sorted(range(len(points)), key=lambda i: values[i])
                points = [points[i] for i in order]
                values = [values[i] for i in order]
                
        return best_x, best_val

    # Strategy:
    # 1. Try a few different initial simplex scales to handle different conditionings
    # 2. Run Nelder-Mead
    # 3. If Nelder-Mead gets close, try a final Newton polish (using numerical derivative carefully)
    
    x0 = float(x0)
    best_x = x0
    best_fx = objective(x0)
    
    if best_fx <= 1e-8:
        return x0
    
    # Try multiple initial scales
    scales = [1e-2, 1e-1, 1.0, 10.0, 100.0]
    
    for scale in scales:
        if time_left() < 5 or eval_count >= MAX_EVALS - 10:
            break
        
        try:
            x_nm, val_nm = nelder_mead(x0, max_iter=150)
            if x_nm is not None and math.isfinite(x_nm) and val_nm < best_fx:
                best_x = x_nm
                best_fx = val_nm
                if best_fx <= 1e-8:
                    return best_x
        except:
            pass
        
        # If we found a good candidate, try to refine with smaller scale
        if best_fx < 1e-4:
             # Restart Nelder-Mead around best_x with smaller scale
             if time_left() > 5 and eval_count < MAX_EVALS - 10:
                 try:
                     x_nm2, val_nm2 = nelder_mead(best_x, max_iter=50)
                     if x_nm2 is not None and math.isfinite(x_nm2) and val_nm2 < best_fx:
                         best_x = x_nm2
                         best_fx = val_nm2
                         if best_fx <= 1e-8:
                             return best_x
                 except:
                     pass
             break # Don't try larger scales if we are already converging
    
    # Final polish: If we are very close, try a few Newton steps to squeeze out precision
    # Newton is fast if we are in the basin of attraction
    if best_fx < 1e-4 and eval_count < MAX_EVALS - 20 and time_left() > 2:
        try:
            x = best_x
            for _ in range(20):
                fx = call_f(x)
                if abs(fx) <= 1e-8:
                    return x
                h = 1e-7 * max(1.0, abs(x))
                d = (call_f(x + h) - fx) / h
                if d == 0 or not math.isfinite(d):
                    break
                step = fx / d
                # Damping
                if abs(step) > 1.0:
                    step = 1.0 * math.copysign(1.0, step)
                x_new = x - step
                if not math.isfinite(x_new):
                    break
                x = x_new
            if math.isfinite(x):
                fx = call_f(x)
                if abs(fx) < best_fx:
                    best_x = x
                    best_fx = abs(fx)
        except:
            pass
            
    return best_x

# EVOLVE-BLOCK-END
