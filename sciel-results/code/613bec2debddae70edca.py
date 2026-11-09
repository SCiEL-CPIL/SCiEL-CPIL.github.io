import math

def find_root(f, x0):
    x = float(x0)
    try:
        fx = f(x)
    except Exception:
        return x

    if not math.isfinite(fx):
        return x
        
    if abs(fx) <= 1e-8:
        return x

    # Adaptive Step Size Search (Hooke-Jeeves style)
    # This approach does not rely on sign changes (bracketing), which makes it
    # robust for roots where f(x) touches zero without crossing, or for highly
    # oscillatory functions where bracketing might be difficult.
    
    # Initial step size
    # Heuristic: small enough to not skip narrow features, large enough to converge.
    step = 1e-3 * max(1.0, abs(x))
    direction = 1.0
    
    min_fx = abs(fx)
    best_x = x
    
    max_iter = 3000
    min_step = 1e-12 * max(1.0, abs(x))
    max_step = 1e6 * max(1.0, abs(x))
    
    for _ in range(max_iter):
        if step < min_step or step > max_step:
            break
            
        x_try = x + direction * step
        
        try:
            fx_try = f(x_try)
        except Exception:
            # If evaluation fails, shrink step and reverse
            step *= 0.5
            direction *= -1
            continue

        if not math.isfinite(fx_try):
            step *= 0.5
            direction *= -1
            continue

        if abs(fx_try) < min_fx:
            # Accept the move
            x = x_try
            fx = fx_try
            min_fx = abs(fx)
            best_x = x
            
            if min_fx <= 1e-8:
                return x
                
            # Increase step size to explore faster
            # Using 1.5 for stability
            step *= 1.5
        else:
            # Reject the move
            # Reverse direction and shrink step
            direction *= -1.0
            step *= 0.5

    return best_x
