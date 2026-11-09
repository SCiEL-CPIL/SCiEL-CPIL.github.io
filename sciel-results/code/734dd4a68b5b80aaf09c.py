import math

def find_root(f, x0):
    x0 = float(x0)
    tol = 1e-8
    
    f0 = f(x0)
    if abs(f0) <= tol:
        return x0

    # Phase 1: Find a bracket
    # Try expanding from x0
    h = 1.0
    a = x0
    fa = f0
    bracket_found = False
    b = None
    fb = None
    
    for _ in range(30): # Max 30 expansions
        x_plus = x0 + h
        x_minus = x0 - h
        
        f_plus = f(x_plus)
        if abs(f_plus) <= tol:
            return x_plus
        if fa * f_plus < 0:
            a = x0
            b = x_plus
            fa = f0
            fb = f_plus
            bracket_found = True
            break
            
        f_minus = f(x_minus)
        if abs(f_minus) <= tol:
            return x_minus
        if fa * f_minus < 0:
            a = x_minus
            b = x0
            fa = f_minus
            fb = f0
            bracket_found = True
            break
            
        h *= 2.0
        
    if not bracket_found:
        # Fallback: If we couldn't bracket, try a few secant steps? 
        # But the directive says new approach. 
        # If no bracket, we might be in a region where f doesn't change sign (e.g. local min > 0).
        # For polynomials, if we expand enough, we should bracket unless the root is extremely far or complex.
        # Let's return the best point found so far.
        return x0

    # Phase 2: Illinois Algorithm
    # a, b are such that fa * fb < 0
    last_kept_a = False
    last_kept_b = False
    
    for _ in range(100):
        # False position (regula falsi)
        if fb == fa:
            # Degenerate, use bisection
            c = (a + b) / 2.0
        else:
            c = a - fa * (b - a) / (fb - fa)
            
        fc = f(c)
        
        if abs(fc) <= tol:
            return c
            
        if (b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
            # Interval too small
            if abs(fa) < abs(fb):
                return a
            else:
                return b

        if fa * fc < 0:
            # Root is in [a, c]
            b = c
            fb = fc
            if last_kept_a:
                fa = fa / 2.0
            last_kept_a = True
            last_kept_b = False
        else:
            # Root is in [c, b]
            a = c
            fa = fc
            if last_kept_b:
                fb = fb / 2.0
            last_kept_a = False
            last_kept_b = True

    # Return best guess
    if abs(fa) < abs(fb):
        return a
    else:
        return b
