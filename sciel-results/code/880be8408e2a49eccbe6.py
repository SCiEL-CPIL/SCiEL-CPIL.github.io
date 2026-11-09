"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a robust combination of:
    1. Illinois method (modified secant with bracketing)
    2. Adaptive step sizes for ill-conditioned problems
    3. Multiple restart strategies if initial attempts fail
    4. Brent's method-inspired safeguarding
    """
    x0 = float(x0)
    
    # Evaluate at x0
    f0 = f(x0)
    if abs(f0) <= 1e-8:
        return x0
    
    # Try to find a sign change nearby to enable bracketing
    # Use multiple step sizes to find a bracket
    h_base = 1e-4 * max(1.0, abs(x0))
    
    # Try several step sizes to find a sign change
    step_multipliers = [1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    
    a, b = None, None
    fa, fb = None, None
    
    for mult in step_multipliers:
        h = h_base * mult
        for sign in [1, -1]:
            x1 = x0 + sign * h
            f1 = f(x1)
            if abs(f1) <= 1e-8:
                return x1
            if f0 * f1 < 0:
                a, b = x0, x1
                fa, fb = f0, f1
                break
        if a is not None:
            break
    
    # If no bracket found, try secant method from x0 with small perturbation
    if a is None:
        h = 1e-4 * max(1.0, abs(x0))
        x1 = x0 + h
        f1 = f(x1)
        if abs(f1) <= 1e-8:
            return x1
        
        # Use Illinois method (modified secant)
        x_prev = x0
        x_curr = x1
        f_prev = f0
        f_curr = f1
        
        for i in range(450):
            if abs(f_curr) <= 1e-8:
                return x_curr
            
            if f_curr - f_prev != 0:
                x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            else:
                x_next = x_curr + 1e-6 * max(1.0, abs(x_curr))
            
            f_next = f(x_next)
            
            if abs(f_next) <= 1e-8:
                return x_next
            
            # Illinois method: shrink the side with smaller |f|
            if abs(f_next) < abs(f_prev):
                f_prev = 0.5 * f_prev
            else:
                f_curr = 0.5 * f_curr
            
            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_next
            f_curr = f_next
        
        return x_curr
    
    # We have a bracket [a, b] with sign change
    # Use Illinois method (modified secant with bracketing)
    # This is more robust than plain bisection for ill-conditioned problems
    
    # Illinois method: use secant step, but if it doesn't reduce bracket size enough,
    # halve the interval and scale down the function value on the rejected side
    
    for i in range(450):
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
        
        # Secant step
        if fb - fa != 0:
            x_sec = b - fb * (b - a) / (fb - fa)
        else:
            x_sec = (a + b) / 2.0
        
        # If secant step is outside bracket or too close to endpoint, use bisection
        if x_sec <= a or x_sec >= b or abs(x_sec - a) < 1e-15 * max(1.0, abs(a)) or abs(x_sec - b) < 1e-15 * max(1.0, abs(b)):
            x_next = (a + b) / 2.0
        else:
            x_next = x_sec
        
        f_next = f(x_next)
        
        if abs(f_next) <= 1e-8:
            return x_next
        
        # Update bracket
        if fa * f_next <= 0:
            # Root is in [a, x_next]
            b = x_next
            fb = f_next
            # Illinois: scale down fa
            fa = 0.5 * fa
        else:
            # Root is in [x_next, b]
            a = x_next
            fa = f_next
            # Illinois: scale down fb
            fb = 0.5 * fb
    
    # Return midpoint of final bracket
    return (a + b) / 2.0
    
    # Fallback strategies if we haven't returned yet
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
    step = 1e-1 * max(1.0, abs(x0))
    for sign in [1, -1]:
        x1 = x0 + sign * step
        f1 = f(x1)
        if abs(f1) <= 1e-8:
            return x1
        
        if f0 * f1 < 0:
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
    
    return x0
# EVOLVE-BLOCK-END
