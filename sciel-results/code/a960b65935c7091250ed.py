"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust scalar root finder for smooth polynomial equations.
    Returns x with |f(x)| <= 1e-8.
    """
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill
    
    x = float(x0)
    
    # Track best solution found so far
    best_x = x
    best_fx = f(x)
    best_abs_fx = abs(best_fx)
    
    # If already close enough
    if best_abs_fx <= 1e-8:
        return best_x
    
    # Phase 1: Try Newton-Raphson with safeguarding
    # Use adaptive step size and backtracking if needed
    for iteration in range(80):
        if time.time() > DEADLINE:
            break
        
        fx = f(x)
        if abs(fx) <= 1e-8:
            return x
        
        # Update best
        if abs(fx) < best_abs_fx:
            best_x = x
            best_abs_fx = abs(fx)
        
        # Compute derivative using finite differences
        h = 1e-8 * max(1.0, abs(x))
        fx_plus = f(x + h)
        fx_minus = f(x - h)
        dfx = (fx_plus - fx_minus) / (2 * h)
        
        if abs(dfx) < 1e-15:
            # Derivative too small, try a different approach
            break
        
        # Newton step
        delta = fx / dfx
        
        # Safeguard: limit step size
        max_step = 10.0 * max(1.0, abs(x))
        if abs(delta) > max_step:
            delta = max_step if delta > 0 else -max_step
        
        x_new = x + delta
        
        # Check for non-finite
        if not math.isfinite(x_new):
            x_new = x - 0.1 * delta if math.isfinite(x - 0.1 * delta) else x
        
        # Evaluate new point
        fx_new = f(x_new)
        
        # Backtracking if function value got worse
        if abs(fx_new) > 2.0 * abs(fx):
            # Try half step
            x_half = x + 0.5 * delta
            fx_half = f(x_half)
            if abs(fx_half) < abs(fx):
                x = x_half
                fx = fx_half
            else:
                # Try quarter step
                x_quarter = x + 0.25 * delta
                fx_quarter = f(x_quarter)
                if abs(fx_quarter) < abs(fx):
                    x = x_quarter
                    fx = fx_quarter
                else:
                    # Stuck, try bisection-like approach
                    break
        else:
            x = x_new
            fx = fx_new
        
        if abs(fx) <= 1e-8:
            return x
    
    # Phase 2: If Newton failed, try a more robust approach
    # Use a combination of bisection and secant method
    if time.time() < DEADLINE:
        # Establish a bracket around x0
        a = x0
        fa = f(a)
        
        # Try to find a bracket by expanding
        step = max(1.0, abs(x0))
        for expand_iter in range(50):
            if time.time() > DEADLINE:
                break
            
            b = a + step
            fb = f(b)
            if fa * fb < 0:
                # Found a bracket, use bisection
                for bisect_iter in range(60):
                    if time.time() > DEADLINE:
                        break
                    mid = (a + b) / 2.0
                    fm = f(mid)
                    if abs(fm) <= 1e-8:
                        return mid
                    if fa * fm < 0:
                        b = mid
                        fb = fm
                    else:
                        a = mid
                        fa = fm
                # Return the midpoint as best estimate
                mid = (a + b) / 2.0
                fm = f(mid)
                if abs(fm) < best_abs_fx:
                    best_x = mid
                    best_abs_fx = abs(fm)
                return best_x
            
            # Try the other direction
            b = a - step
            fb = f(b)
            if fa * fb < 0:
                # Found a bracket, use bisection
                for bisect_iter in range(60):
                    if time.time() > DEADLINE:
                        break
                    mid = (a + b) / 2.0
                    fm = f(mid)
                    if abs(fm) <= 1e-8:
                        return mid
                    if fa * fm < 0:
                        b = mid
                        fb = fm
                    else:
                        a = mid
                        fa = fm
                mid = (a + b) / 2.0
                fm = f(mid)
                if abs(fm) < best_abs_fx:
                    best_x = mid
                    best_abs_fx = abs(fm)
                return best_x
            
            step *= 2.0
            if step > 1e6:
                break
    
    # Return the best solution found
    return best_x
# EVOLVE-BLOCK-END
