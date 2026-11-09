"""Root finder using Illinois method with Newton polish for oscillatory functions."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Strategy:
      1. Use Illinois/secant-like method to bracket and converge.
      2. Polish with Newton's method for high precision.
    """
    x0 = float(x0)
    start_time = time.time()
    deadline = start_time + 110.0  # 110s margin out of 120s limit
    
    # First evaluate at x0
    fx0 = f(x0)
    if abs(fx0) <= 1e-8:
        return x0
    
    # Try to find a bracket around x0
    # We'll use a secant/Illinois approach
    # Start with two points: x0 and a small step
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h
    fx1 = f(x1)
    
    # If both same sign, try expanding
    if fx0 * fx1 > 0:
        # Try other direction
        x1 = x0 - h
        fx1 = f(x1)
        if fx0 * fx1 > 0:
            # Expand further
            h = 1e-2 * max(1.0, abs(x0))
            x1 = x0 + h
            fx1 = f(x1)
            if fx0 * fx1 > 0:
                x1 = x0 - h
                fx1 = f(x1)
                if fx0 * fx1 > 0:
                    # Just use secant from x0
                    pass
    
    # Illinois method
    a, fa = x0, fx0
    b, fb = x1, fx1
    
    # Ensure we have a bracket if possible
    if fa * fb > 0:
        # No bracket, use secant method directly
        for i in range(80):
            if time.time() > deadline:
                break
            if fb == 0:
                return b
            if fa == 0:
                return a
            if abs(fb) <= 1e-8:
                return b
            if abs(fa) <= 1e-8:
                return a
            
            # Secant step
            denom = fb - fa
            if abs(denom) < 1e-30:
                break
            x_new = b - fb * (b - a) / denom
            fx_new = f(x_new)
            
            # Update bracket if we have one
            if fa * fb > 0:
                # No bracket, just track best
                if abs(fx_new) < abs(fb):
                    a, fa = b, fb
                    b, fb = x_new, fx_new
                else:
                    a, fa = b, fb
                    b, fb = x_new, fx_new
            else:
                # We have a bracket
                if fa * fx_new <= 0:
                    b, fb = x_new, fx_new
                else:
                    # Shrink the side that had the larger function value
                    if abs(fa) > abs(fb):
                        a, fa = b, fb
                    b, fb = x_new, fx_new
                    # Illinois: shrink the side we're replacing
                    if fa * fb <= 0:
                        # Keep bracket, shrink a side
                        if abs(fa) > abs(fb):
                            a = a + (b - a) * 0.5
                            fa = f(a)
                        else:
                            b = a + (b - a) * 0.5
                            fb = f(b)
        
        # Return best point
        if abs(fa) <= abs(fb):
            return a
        else:
            return b
    
    # We have a bracket [a, b]
    for i in range(100):
        if time.time() > deadline:
            break
        if fb == 0:
            return b
        if fa == 0:
            return a
        if abs(fb) <= 1e-8:
            return b
        if abs(fa) <= 1e-8:
            return a
        
        # Secant step
        denom = fb - fa
        if abs(denom) < 1e-30:
            # Bisection fallback
            x_new = (a + b) / 2
            fx_new = f(x_new)
        else:
            x_new = b - fb * (b - a) / denom
            fx_new = f(x_new)
        
        # Check if x_new is within bracket
        if not (a < x_new < b or b < x_new < a):
            x_new = (a + b) / 2
            fx_new = f(x_new)
        
        if fa * fx_new <= 0:
            b, fb = x_new, fx_new
        else:
            a, fa = x_new, fx_new
            # Illinois: shrink the side we just replaced
            if abs(fa) > abs(fb):
                a = a + (b - a) * 0.5
                fa = f(a)
        
        # Check convergence
        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
            break
    
    # Return the better of a or b
    if abs(fa) <= abs(fb):
        result = a
    else:
        result = b
    
    # Newton polish with tighter tolerance and more iterations
    for i in range(30):
        if time.time() > deadline:
            break
        fx = f(result)
        if abs(fx) <= 1e-10:
            return result
        h = 1e-7 * max(1.0, abs(result))
        d = (f(result + h) - fx) / h
        if abs(d) < 1e-30:
            break
        result = result - fx / d
    
    return result
# EVOLVE-BLOCK-END
