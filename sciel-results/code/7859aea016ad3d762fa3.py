"""IQI-based scalar root finder exploiting polynomial structure."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses inverse quadratic interpolation for polynomial roots,
    with secant fallback. Minimal call count per step.
    """
    DEADLINE = 110.0
    START_TIME = time.time()
    TOL = 1e-8
    
    def elapsed():
        return time.time() - START_TIME
    
    def check_deadline():
        return elapsed() < DEADLINE
    
    def is_finite(x):
        return math.isfinite(x)
    
    # Initial evaluations
    x1 = float(x0)
    try:
        f1 = f(x1)
        if not is_finite(f1):
            f1 = None
    except Exception:
        f1 = None
    
    if f1 is None:
        # Try a small perturbation
        x1 = x1 + 1e-4
        try:
            f1 = f(x1)
            if not is_finite(f1):
                f1 = None
        except Exception:
            f1 = None
        if f1 is None:
            return x1
    
    if abs(f1) <= TOL:
        return x1
    
    # Get a second point for secant
    h = 1e-6 * max(1.0, abs(x1))
    x2 = x1 + h
    try:
        f2 = f(x2)
        if not is_finite(f2):
            f2 = None
    except Exception:
        f2 = None
    
    if f2 is None:
        # Try other perturbation
        x2 = x1 - h
        try:
            f2 = f(x2)
            if not is_finite(f2):
                f2 = None
        except Exception:
            f2 = None
        if f2 is None:
            # Just do bisection-like from x1
            for _ in range(50):
                if not check_deadline():
                    break
                # Can't do much without second point; return best
                break
            return x1
    
    if abs(f2) <= TOL:
        return x2
    
    # Main loop: IQI when 3 points available, secant otherwise
    best_x = x1 if abs(f1) < abs(f2) else x2
    best_f = min(abs(f1), abs(f2))
    
    for it in range(60):
        if not check_deadline():
            break
        
        if abs(f1) <= TOL:
            return x1
        if abs(f2) <= TOL:
            return x2
        
        # Try to get a third point for IQI
        # Use secant step to get x3
        if f2 != f1:
            x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
        else:
            # Degenerate: use Newton-like step
            d = (f2 - f1) / (x2 - x1)
            if d == 0 or not is_finite(d):
                break
            x3 = x2 - f2 / d
        
        if not is_finite(x3):
            break
        
        try:
            f3 = f(x3)
            if not is_finite(f3):
                f3 = None
        except Exception:
            f3 = None
        
        if f3 is None:
            break
        
        if abs(f3) <= TOL:
            return x3
        
        # Update best
        if abs(f3) < best_f:
            best_f = abs(f3)
            best_x = x3
        
        # Now we have 3 points: (x1,f1), (x2,f2), (x3,f3)
        # Inverse quadratic interpolation: fit x = g(f) as quadratic
        # through (f1,x1), (f2,x2), (f3,x3) and find f=0
        # g(f) = a*f^2 + b*f + c
        # Using Lagrange form:
        # x_new = x1 * f2*f3 / ((f1-f2)*(f1-f3)) 
        #       + x2 * f1*f3 / ((f2-f1)*(f2-f3))
        #       + x3 * f1*f2 / ((f3-f1)*(f3-f2))
        
        denom1 = (f1 - f2) * (f1 - f3)
        denom2 = (f2 - f1) * (f2 - f3)
        denom3 = (f3 - f1) * (f3 - f2)
        
        if abs(denom1) > 1e-300 and abs(denom2) > 1e-300 and abs(denom3) > 1e-300:
            x_new = (x1 * f2 * f3 / denom1 + 
                     x2 * f1 * f3 / denom2 + 
                     x3 * f1 * f2 / denom3)
        else:
            # Fallback to secant
            if f2 != f1:
                x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
            elif f3 != f2:
                x_new = x3 - f3 * (x3 - x2) / (f3 - f2)
            else:
                break
        
        if not is_finite(x_new):
            break
        
        # Safeguard: if x_new is too far, use secant instead
        # Check if it's reasonable (not wildly diverging)
        scale = max(1.0, abs(x1), abs(x2), abs(x3))
        if abs(x_new - x2) > 10 * scale:
            # Use secant step instead
            if f2 != f1:
                x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
            elif f3 != f2:
                x_new = x3 - f3 * (x3 - x2) / (f3 - f2)
            else:
                break
        
        if not is_finite(x_new):
            break
        
        # Shift points: keep the two closest to root
        # Keep the two points with smallest |f|
        pts = [(x1, f1), (x2, f2), (x3, f3)]
        pts.sort(key=lambda p: abs(p[1]))
        x1, f1 = pts[0]
        x2, f2 = pts[1]
        # x3, f3 = pts[2]  # discard the worst
        
        # If x_new is very close to an existing point, nudge
        if abs(x_new - x1) < 1e-15 * max(1.0, abs(x1)):
            x_new = x1 + 1e-12 * max(1.0, abs(x1))
        if abs(x_new - x2) < 1e-15 * max(1.0, abs(x2)):
            x_new = x2 + 1e-12 * max(1.0, abs(x2))
    
    # Final check
    if abs(f1) <= TOL:
        return x1
    if abs(f2) <= TOL:
        return x2
    
    # Return best found
    return best_x
# EVOLVE-BLOCK-END
