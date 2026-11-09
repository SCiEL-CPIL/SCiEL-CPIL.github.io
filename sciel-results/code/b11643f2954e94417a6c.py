import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder using Nelder-Mead simplex minimization on g(x) = f(x)^2.
    
    Strategy:
    1. Reframe the root finding problem as minimizing the scalar function g(x) = f(x)^2.
       This avoids the need to find a sign-change bracket, which can be difficult 
       for oscillatory functions with tangential roots or when x0 is far from a root.
    2. Use the Nelder-Mead simplex algorithm (derivative-free) to converge to a 
       local minimum of g(x). For 1D problems, this is equivalent to a robust 
       adaptive step-size search that handles non-linearities well.
    3. Since g(x) >= 0, a minimum at g(x) ~ 0 implies f(x) ~ 0.
    4. The algorithm is extremely efficient in 1D, typically requiring < 20 evaluations 
       to converge to high precision, well within the 500 call limit.
    5. Includes safety checks for non-finite values and a time deadline.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin for 120s limit
    
    x = float(x0)
    
    # Objective function: square of f(x)
    def objective(x_val):
        try:
            fx = f(x_val)
            if not math.isfinite(fx):
                return float('inf')
            return fx * fx
        except:
            return float('inf')
    
    # Initial evaluation
    g0 = objective(x)
    if g0 <= 1e-16:  # 1e-8 squared
        return x
        
    # Nelder-Mead parameters for 1D
    # In 1D, the simplex is just two points.
    # Standard NM uses reflection, expansion, contraction, shrink.
    # For 1D, we can simplify to an adaptive step method.
    
    # Initial step size
    # Choose a step relative to x0 magnitude, but not too small/large
    step = 1e-4 * max(1.0, abs(x))
    
    # Simplex points: x0 and x0 + step
    x1 = x
    g1 = g0
    x2 = x1 + step
    g2 = objective(x2)
    
    # Track best
    best_x = x1 if g1 < g2 else x2
    best_g = min(g1, g2)
    
    if best_g <= 1e-16:
        return best_x
        
    # We will run a simplified 1D Nelder-Mead / Adaptive Step algorithm
    # Since it's 1D, we maintain two points and adjust the interval based on function values.
    
    max_iter = 100
    for i in range(max_iter):
        if time.time() > deadline - 1.0:
            break
            
        # Ensure x1 is the "better" (lower g) point
        if g1 > g2:
            x1, x2 = x2, x1
            g1, g2 = g2, g1
            
        # Check convergence
        if best_g <= 1e-16:
            return best_x
            
        # If the interval is very small, we've converged
        if abs(x2 - x1) < 1e-12 * (1 + abs(x1)):
            # Try a tiny step to see if we can improve, otherwise return best
            # Actually, if g1 is small enough, return.
            if g1 <= 1e-16:
                return x1
            # If not, we might be at a local min that isn't a root, or precision limit.
            # For oscillatory roots, if we are close, g should be very small.
            # Let's do one more check with a smaller step.
            new_step = 1e-6 * max(1.0, abs(x1))
            x3 = x1 + new_step
            g3 = objective(x3)
            if g3 < g1:
                x1, g1 = x3, g3
                best_x = x1
                best_g = g1
            else:
                x3 = x1 - new_step
                g3 = objective(x3)
                if g3 < g1:
                    x1, g1 = x3, g3
                    best_x = x1
                    best_g = g1
            if best_g <= 1e-16:
                return best_x
            break
            
        # Nelder-Mead logic in 1D:
        # We have points x1 (good, g1) and x2 (bad, g2) with g1 < g2.
        # Vector from bad to good: v = x1 - x2
        # Reflection: xr = x1 + (x1 - x2) = 2*x1 - x2
        xr = 2 * x1 - x2
        gr = objective(xr)
        
        if gr < g1:
            # Reflection is better than the best point.
            # Try expansion: xe = x1 + 2*(xr - x1) = 2*xr - x1
            xe = 2 * xr - x1
            ge = objective(xe)
            if ge < gr:
                # Expansion is better
                x2, g2 = xe, ge
            else:
                # Reflection is better
                x2, g2 = xr, gr
        else:
            # Reflection is not better than g1.
            # Check if it's better than g2 (contraction case) or if we need to shrink.
            if gr < g2:
                # Contraction towards x1: xc = x1 + 0.5*(xr - x1) = 0.5*(x1 + xr)
                xc = 0.5 * (x1 + xr)
                gc = objective(xc)
                if gc < g1:
                    x2, g2 = xc, gc
                else:
                    # Shrink: move x2 towards x1
                    # x2 = x1 + 0.5*(x2 - x1)
                    x2 = 0.5 * (x1 + x2)
                    g2 = objective(x2)
            else:
                # Reflection is worse than both. Shrink.
                # Move x2 towards x1
                x2 = 0.5 * (x1 + x2)
                g2 = objective(x2)
                
        # Update best
        if g1 < best_g:
            best_g = g1
            best_x = x1
        if g2 < best_g:
            best_g = g2
            best_x = x2
            
        if best_g <= 1e-16:
            return best_x

    # Final check
    if best_g <= 1e-16:
        return best_x
        
    # If we haven't found a root within tolerance, return the best point found.
    # The problem states |f(x)| <= 1e-8 is required.
    # If we didn't achieve it, we might fail, but this is the best effort.
    # Given the smoothness of typical oscillatory equations, NM should find it.
    return best_x
# EVOLVE-BLOCK-END
