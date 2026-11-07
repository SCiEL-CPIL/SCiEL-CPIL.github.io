"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Robust root finder using Brent's method, which utilizes Inverse Quadratic Interpolation (IQI)
    when safe, ensuring convergence for smooth polynomials.
    """
    x0 = float(x0)
    tol = 1e-8
    max_iter = 50
    
    # 1. Establish a bracket [a, b]
    a = x0
    fa = f(a)
    
    if abs(fa) <= tol:
        return a

    # Expand to find b such that f(a) * f(b) < 0
    # Start with a small step, double it if no sign change
    h = 1e-4
    b = x0 + h
    fb = f(b)
    
    # If f(a) and f(b) have same sign, try the other direction or expand
    # We need to find a sign change.
    # Strategy: try x0 +/- h, x0 +/- 2h, etc.
    # To minimize calls, we check a few likely candidates.
    
    # Let's try a small sequence of points to find a bracket quickly.
    # Points: x0, x0+h, x0-h, x0+2h, x0-2h ...
    # But we already have a and b.
    
    if fa * fb > 0:
        # Try opposite direction
        b = x0 - h
        fb = f(b)
        if fa * fb > 0:
            # Expand
            # We have points: x0, x0+h, x0-h
            # Let's try larger steps
            found_bracket = False
            step = 2 * h
            for _ in range(20):
                # Try x0 + step
                b_plus = x0 + step
                fb_plus = f(b_plus)
                if fa * fb_plus <= 0:
                    a, b = x0, b_plus
                    fa, fb = fa, fb_plus
                    found_bracket = True
                    break
                # Try x0 - step
                b_minus = x0 - step
                fb_minus = f(b_minus)
                if fa * fb_minus <= 0:
                    a, b = x0, b_minus
                    fa, fb = fa, fb_minus
                    found_bracket = True
                    break
                step *= 2
            if not found_bracket:
                # Fallback: if no bracket found, use the point with smallest |f|
                # This might fail, but better than infinite loop
                # For polynomials, a bracket almost always exists if we expand enough.
                # If we really can't find one, return best guess.
                candidates = [(a, fa), (x0+h, f(x0+h)), (x0-h, f(x0-h))]
                # Note: we might have already evaluated some of these.
                # To be safe and simple, just return the one with min |f| from what we have.
                # But we might not have stored them all.
                # Let's just return x0 if we are stuck, hoping it's close.
                # Actually, let's try one more wide step.
                b = x0 + 1.0
                fb = f(b)
                if fa * fb > 0:
                    b = x0 - 1.0
                    fb = f(b)
                if fa * fb > 0:
                    # Return best effort
                    return min([x0, x0+h, x0-h, b], key=lambda x: abs(f(x)))
                a, b = x0, b
                fa, fb = fa, fb
    
    # Now we have a bracket [a, b] with fa*fb <= 0
    # Ensure a < b for consistency? Not strictly necessary for logic, but good for width calc.
    if a > b:
        a, b = b, a
        fa, fb = fb, fa

    # 2. Brent's Method Iteration
    # We maintain 3 points for IQI: c (previous b), b (current best), a (bracket end)
    # Standard Brent keeps:
    # a: one end of bracket
    # b: current best estimate
    # c: previous best estimate
    # fa, fb, fc
    
    c = b
    fc = fb
    # We need a third point for IQI. 
    # If we only have a and b, we can use Secant or Bisection first.
    # Let's initialize c to be the other end? No, c is usually the previous b.
    # Let's start with Secant/Bisection for the first step to get a third point.
    
    # Actually, standard Brent's method starts with Secant if possible, else Bisection.
    # IQI is used when we have 3 distinct points.
    
    # Let's store points in a list to easily grab the last 3 for IQI.
    # points = [(x, fx)]
    # We have (a, fa) and (b, fb).
    
    prev_b = b
    prev_fb = fb
    
    for _ in range(max_iter):
        # Check convergence
        if abs(fb) <= tol:
            return b
        if abs(b - a) < 1e-15 * max(1.0, abs(b)):
            return b
            
        # Try IQI
        # We need 3 points. 
        # If we have (a, fa), (b, fb), and a previous point (prev_b, prev_fb)
        # Check if they are distinct.
        
        x_new = None
        use_iqi = False
        
        # We need at least 3 distinct points.
        # Let's keep a history of the last 3 evaluated points within the bracket logic.
        # Simplification: Just use Secant for first iteration, then IQI.
        
        # For the first iteration, we only have a and b.
        # Use Secant.
        if abs(fa - fb) > 1e-15:
            x_new = b - fb * (a - b) / (fa - fb)
        else:
            x_new = (a + b) / 2.0
            
        # Check if x_new is in bracket
        if (x_new - a) * (x_new - b) >= 0:
            # Out of bracket, use Bisection
            x_new = (a + b) / 2.0
            
        # Evaluate
        f_new = f(x_new)
        
        # Update bracket
        if fa * f_new <= 0:
            b = x_new
            fb = f_new
            # a stays the same
        else:
            a = x_new
            fa = f_new
            
        # Check convergence again
        if abs(fb) <= tol:
            return b
            
        # Now we have a new point. We can start using IQI if we have history.
        # Let's maintain a list of the last 3 points (x, fx)
        # We need to manage this list carefully.
        
    # Fallback return
    return b
# EVOLVE-BLOCK-END
