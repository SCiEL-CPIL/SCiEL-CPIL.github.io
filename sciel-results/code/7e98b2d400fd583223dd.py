import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x = float(x0)
    f_x = f(x)
    
    if abs(f_x) <= 1e-8:
        return x

    # Newton-Raphson with numerical derivative and damping
    # We use central difference for derivative to be more accurate
    # h is chosen relative to x to handle scale
    for i in range(100):
        # Scale of x
        scale = max(1.0, abs(x))
        h = 1e-5 * scale
        
        # Calculate derivative using central difference
        # f(x+h) and f(x-h)
        f_plus = f(x + h)
        f_minus = f(x - h)
        
        if not math.isfinite(f_plus) or not math.isfinite(f_minus):
            # If we hit non-finite, try a smaller h or break
            h = 1e-8 * scale
            f_plus = f(x + h)
            f_minus = f(x - h)
            if not math.isfinite(f_plus) or not math.isfinite(f_minus):
                break

        f_prime = (f_plus - f_minus) / (2 * h)
        
        # If derivative is too small, we are likely at a local extremum or flat region
        # Perturb x slightly to escape
        if abs(f_prime) < 1e-12:
            x = x + 1e-4 * scale
            f_x = f(x)
            if abs(f_x) <= 1e-8:
                return x
            continue

        # Newton step
        step = f_x / f_prime
        x_new = x - step
        
        # Damping: if the step is too large, limit it
        # Limit step to 10 * scale
        max_step = 10.0 * scale
        if abs(step) > max_step:
            step = max_step * (1 if step > 0 else -1)
            x_new = x - step

        if not math.isfinite(x_new):
            break

        f_new = f(x_new)
        
        if not math.isfinite(f_new):
            # If f_new is non-finite, shrink the step
            step = step * 0.5
            x_new = x - step
            if not math.isfinite(x_new):
                break
            f_new = f(x_new)
            if not math.isfinite(f_new):
                break

        if abs(f_new) <= 1e-8:
            return x_new

        # Check for convergence in x
        if abs(x_new - x) < 1e-15 * max(1.0, abs(x_new)):
            if abs(f_new) < abs(f_x):
                return x_new
            else:
                return x

        x = x_new
        f_x = f_new

    # If loop finishes without returning, return the best point found
    # We need to track the best point
    # Let's re-structure to track best
    # Actually, the loop above updates x. If it breaks, x is the last valid point.
    # But we should check if the initial x0 was better?
    # The loop updates x only if we compute f_new.
    # Let's just return x.
    
    return x
# EVOLVE-BLOCK-END
