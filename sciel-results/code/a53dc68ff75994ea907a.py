"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a robust combination of:
    1. Secant method (no derivative needed, faster than bisection)
    2. Bisection fallback when sign change is detected
    3. Adaptive step sizes for ill-conditioned problems
    4. Multiple restart strategies if initial attempts fail
    """
    x0 = float(x0)
    
    # Try to find a root using a hybrid secant/bisection approach
    # First, evaluate at x0
    f0 = f(x0)
    if abs(f0) <= 1e-8:
        return x0
    
    # Try to find a sign change nearby to enable bisection
    # Also use secant method as primary
    
    # Strategy 1: Secant method with careful step selection
    # Choose a second point to start secant
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h
    f1 = f(x1)
    
    if abs(f1) <= 1e-8:
        return x1
    
    # If we have a sign change, we can use bisection as a fallback
    # Otherwise, use secant
    
    x_a, f_a = x0, f0
    x_b, f_b = x1, f1
    
    # Check for sign change
    has_sign_change = (f_a * f_b < 0)
    
    # Use secant method primarily, with bisection safeguard
    x_prev = x_a
    x_curr = x_b
    f_prev = f_a
    f_curr = f_b
    
    for i in range(490):  # Leave some evaluations for fallback
        if abs(f_curr) <= 1e-8:
            return x_curr
        
        # Secant step
        if f_curr - f_prev != 0:
            x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        else:
            # Derivative is zero, try a different step
            x_next = x_curr + 1e-6 * max(1.0, abs(x_curr))
        
        # If we have a sign change, check if x_next is within the bracket
        if has_sign_change:
            # Ensure x_next stays within the bracket or update bracket
            if (f_a * f_b < 0):
                # Standard bisection-like safeguard
                if (x_next < min(x_a, x_b)) or (x_next > max(x_a, x_b)):
                    # Use bisection instead
                    x_next = (x_a + x_b) / 2.0
        
        f_next = f(x_next)
        
        if abs(f_next) <= 1e-8:
            return x_next
        
        # Update bracket if we have a sign change
        if has_sign_change:
            if f_a * f_next <= 0:
                x_b = x_next
                f_b = f_next
            else:
                x_a = x_next
                f_a = f_next
            # Update has_sign_change if needed
            if f_a * f_b > 0:
                has_sign_change = False
        
        # Update for secant
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_next
        f_curr = f_next
    
    # If we haven't found a root yet, try alternative strategies
    # Strategy 2: Try different initial perturbations
    perturbations = [
        1e-3 * max(1.0, abs(x0)),
        -1e-3 * max(1.0, abs(x0)),
        1e-2 * max(1.0, abs(x0)),
        -1e-2 * max(1.0, abs(x0)),
        1e-5 * max(1.0, abs(x0)),
        -1e-5 * max(1.0, abs(x0)),
    ]
    
    for h in perturbations:
        x1 = x0 + h
        f1 = f(x1)
        if abs(f1) <= 1e-8:
            return x1
        
        # Try secant from x0 and x1
        x_prev = x0
        x_curr = x1
        f_prev = f0
        f_curr = f1
        
        for i in range(50):
            if abs(f_curr) <= 1e-8:
                return x_curr
            
            if f_curr - f_prev != 0:
                x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            else:
                break
            
            f_next = f(x_next)
            
            if abs(f_next) <= 1e-8:
                return x_next
            
            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_next
            f_curr = f_next
    
    # Strategy 3: Bisection with wider bracket
    # Try to find a sign change with larger steps
    step = 1e-1 * max(1.0, abs(x0))
    for sign in [1, -1]:
        x1 = x0 + sign * step
        f1 = f(x1)
        if abs(f1) <= 1e-8:
            return x1
        
        if f0 * f1 < 0:
            # Bisection
            a, b = x0, x1
            fa, fb = f0, f1
            for i in range(60):
                mid = (a + b) / 2.0
                fm = f(mid)
                if abs(fm) <= 1e-8:
                    return mid
                if fa * fm <= 0:
                    b = mid
                    fb = fm
                else:
                    a = mid
                    fa = fm
            return (a + b) / 2.0
    
    # Return best estimate
    return x_curr
# EVOLVE-BLOCK-END
