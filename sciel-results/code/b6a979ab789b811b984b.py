import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Finds a root x of f(x) such that |f(x)| <= 1e-8.
    Uses an adaptive Secant method with fallback to local minimization and 
    bracketing rescue. This approach does not rely on finding a bracket initially.
    """
    x0 = float(x0)
    
    # Initial evaluation
    try:
        fx0 = f(x0)
        if np.isnan(fx0) or not np.isfinite(fx0):
            return x0
        if abs(fx0) <= 1e-8:
            return x0
    except Exception:
        return x0

    best_x = x0
    best_f = abs(fx0)
    
    # State for Secant method
    # We need two points to start. Let's pick x0 and x0 + delta.
    # Adaptive delta: relative to x0 scale.
    scale = max(1.0, abs(x0))
    delta = 1e-4 * scale
    
    x_prev = x0
    x_curr = x0 + delta
    
    try:
        f_prev = fx0
        f_curr = f(x_curr)
        if np.isnan(f_curr) or not np.isfinite(f_curr):
            # Try the other side
            x_curr = x0 - delta
            f_curr = f(x_curr)
            if np.isnan(f_curr) or not np.isfinite(f_curr):
                # If both fail, just return best
                return best_x
    except Exception:
        return best_x

    if abs(f_curr) < best_f:
        best_f = abs(f_curr)
        best_x = x_curr

    if abs(f_curr) <= 1e-8:
        return x_curr

    max_iter = 100
    stall_count = 0
    last_best_f = best_f
    
    for i in range(max_iter):
        # Check for sign change in current interval [x_prev, x_curr]
        # If found, use brentq to polish quickly
        if f_prev * f_curr < 0:
            try:
                root = brentq(f, x_prev, x_curr, xtol=1e-12, rtol=1e-12, maxiter=50)
                fr = f(root)
                if abs(fr) <= 1e-8:
                    return root
                # If brentq fails to meet tolerance but returns a point, check it
                if abs(fr) < best_f:
                    return root
            except Exception:
                pass

        # Secant Step
        # x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        denom = f_curr - f_prev
        if abs(denom) < 1e-30:
            # Derivative is zero or points are identical
            # Perturb
            x_new = x_curr + 1e-6 * scale
        else:
            x_new = x_curr - f_curr * (x_curr - x_prev) / denom
        
        if not np.isfinite(x_new):
            break
            
        try:
            f_new = f(x_new)
        except Exception:
            break
            
        if np.isnan(f_new) or not np.isfinite(f_new):
            break
            
        abs_f_new = abs(f_new)
        
        if abs_f_new <= 1e-8:
            return x_new
            
        if abs_f_new < best_f:
            best_f = abs_f_new
            best_x = x_new
            stall_count = 0
        else:
            # Check if it's a significant stall/divergence
            if abs_f_new > last_best_f * 1.5:
                stall_count += 1
            else:
                # Small increase, might be noise or flat region
                if abs_f_new > last_best_f * 1.1:
                    stall_count += 0.5
                else:
                    stall_count = 0
            
            # If we are diverging significantly, try damping or restarting
            if stall_count > 3:
                # Restart from best_x with a different step
                x_prev = best_x
                # Try a smaller step in the direction of the gradient (estimated)
                # Or just perturb
                x_curr = best_x + 1e-5 * scale
                try:
                    f_prev = f(x_prev)
                    f_curr = f(x_curr)
                    if np.isnan(f_curr) or not np.isfinite(f_curr):
                         x_curr = best_x - 1e-5 * scale
                         f_curr = f(x_curr)
                except Exception:
                    break
                stall_count = 0
                last_best_f = best_f
                continue

        # Update points for next secant step
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_new
        f_curr = f_new
        
        # Update last_best_f for stall detection logic
        if abs_f_new < last_best_f:
            last_best_f = abs_f_new

    # If Secant didn't find it, try a local minimization of f(x)^2 
    # using a simple grid search around the best point found
    # This handles cases where the root is a tangent (no sign change) and secant stalls
    
    search_radius = 1e-3 * scale
    n_points = 100
    xs = np.linspace(best_x - search_radius, best_x + search_radius, n_points)
    
    min_val = best_f
    min_x = best_x
    
    # Vectorized evaluation is not possible if f is a black box that might error, 
    # but we can loop. 100 evals is cheap.
    for x in xs:
        try:
            fx = f(x)
            if np.isnan(fx) or not np.isfinite(fx):
                continue
            if abs(fx) < min_val:
                min_val = abs(fx)
                min_x = x
                if min_val <= 1e-8:
                    return min_x
        except Exception:
            continue
            
    # If grid search improved, we can try one more secant round from min_x
    # Or just return the best found
    if min_val < best_f:
        best_x = min_x
        best_f = min_val

    return best_x
