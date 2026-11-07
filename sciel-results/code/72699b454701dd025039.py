"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Find a root of f near x0 using inverse quadratic interpolation (IQI)
    with safeguards. Returns x such that |f(x)| <= 1e-8.
    """
    x = float(x0)
    
    # First evaluate at x0
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # We need at least 3 points for IQI. Start with a bisection-like expansion
    # to get a bracket, then switch to IQI.
    
    # Try to find a bracket [a, b] with f(a) and f(b) having opposite signs
    # Start with a small step and expand
    a = x
    fa = fx
    step = 1.0
    b = None
    fb = None
    
    max_expansions = 50
    for _ in range(max_expansions):
        b = a + step
        fb = f(b)
        if abs(fb) <= 1e-8:
            return b
        if fa * fb < 0:
            break
        # Expand
        step *= 2.0
        if not math.isfinite(b):
            break
    
    # If we found a bracket, use IQI within it
    if b is not None and fa * fb < 0 and math.isfinite(b):
        # We have a bracket [a, b]. Need a third point for IQI.
        # Use the midpoint as the third point.
        c = (a + b) / 2.0
        fc = f(c)
        if abs(fc) <= 1e-8:
            return c
        
        # Now we have three points: (a, fa), (b, fb), (c, fc)
        # Use IQI to find the next point
        for _ in range(100):
            # Ensure we have three distinct points with finite values
            # Pick the three points that give the best IQI estimate
            # Sort by x
            pts = [(a, fa), (b, fb), (c, fc)]
            pts.sort(key=lambda p: p[0])
            
            x1, f1 = pts[0]
            x2, f2 = pts[1]
            x3, f3 = pts[2]
            
            # Inverse quadratic interpolation:
            # Find x such that q(x) = 0, where q is the quadratic
            # interpolant of 1/f at the three points.
            
            # Standard IQI formula:
            # x_new = x1 * (f2*f3)/((f1-f2)*(f1-f3)) 
            #        + x2 * (f1*f3)/((f2-f1)*(f2-f3))
            #        + x3 * (f1*f2)/((f3-f1)*(f3-f2))
            
            denom1 = (f1 - f2) * (f1 - f3)
            denom2 = (f2 - f1) * (f2 - f3)
            denom3 = (f3 - f1) * (f3 - f2)
            
            x_new = None
            if abs(denom1) > 1e-30 and abs(denom2) > 1e-30 and abs(denom3) > 1e-30:
                term1 = x1 * (f2 * f3) / denom1
                term2 = x2 * (f1 * f3) / denom2
                term3 = x3 * (f1 * f2) / denom3
                x_new = term1 + term2 + term3
            
            # Check if x_new is valid and within bracket
            valid = False
            if x_new is not None and math.isfinite(x_new):
                if (a < x_new < b) or (b < x_new < a):
                    valid = True
            
            if not valid:
                # Fall back to bisection
                x_new = (a + b) / 2.0
            
            fx_new = f(x_new)
            if abs(fx_new) <= 1e-8:
                return x_new
            
            # Update bracket
            if fa * fx_new < 0:
                b = x_new
                fb = fx_new
            else:
                a = x_new
                fa = fx_new
            
            # Update the three points: keep the most recent three
            # We want points that are well-distributed. Keep a, b, and x_new
            # but ensure we have three distinct points.
            # Strategy: replace the point that is farthest from the new one
            # or simply keep (a, b, x_new) as the three points for next iteration.
            c = x_new
            fc = fx_new
            
            # Check convergence
            if abs(b - a) < 1e-12:
                # Return the point with smaller |f|
                if abs(fa) < abs(fb):
                    return a
                else:
                    return b
    
    # If no bracket found, try a local minimization approach or just return best
    # Try a few more steps with secant-like method
    x1 = x
    f1 = fx
    x2 = x + 1e-4 * max(1.0, abs(x))
    f2 = f(x2)
    
    for _ in range(100):
        if abs(f2) <= 1e-8:
            return x2
        if abs(f1) <= 1e-8:
            return x1
        if f1 == f2:
            break
        # Secant step
        x_new = x1 - f1 * (x2 - x1) / (f2 - f1)
        if not math.isfinite(x_new):
            break
        f_new = f(x_new)
        if abs(f_new) <= 1e-8:
            return x_new
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new
    
    # Return the best we have
    if abs(fx) <= abs(f1):
        return x
    elif abs(f1) <= abs(f2):
        return x1
    else:
        return x2
# EVOLVE-BLOCK-END
