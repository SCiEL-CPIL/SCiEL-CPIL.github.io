"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Derivative-free root finder using inverse quadratic interpolation (IQI)
    with safeguards for robustness.
    
    Strategy:
    1. Start with x0.
    2. Use a small step to get a second point.
    3. Use IQI (inverse quadratic interpolation) to predict the next root.
    4. If IQI step is too large or goes out of bounds, fall back to bisection-like
       or secant step.
    5. Maintain three most recent points (x, f(x)) for IQI.
    6. Stop when |f(x)| <= 1e-8 (with a tight threshold for safety).
    """
    x = float(x0)
    fx = f(x)
    
    # If already at root
    if abs(fx) <= 1e-12:
        return x
    
    # Get a second point with a small step
    # Choose step size adaptively
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    f1 = f(x1)
    
    if abs(f1) <= 1e-12:
        return x1
    
    # Now we have two points: (x, fx), (x1, f1)
    # For IQI we need three points. Get a third point.
    # Try a secant step from the two points to get a third evaluation.
    if fx == f1:
        # Degenerate, try a different step
        h = 1e-3 * max(1.0, abs(x))
        x1 = x + h
        f1 = f(x1)
        if abs(f1) <= 1e-12:
            return x1
    
    # Secant step to get third point
    x2 = x - fx * (x1 - x) / (f1 - fx)
    f2 = f(x2)
    
    if abs(f2) <= 1e-12:
        return x2
    
    # Now we have three points: (x, fx), (x1, f1), (x2, f2)
    # Keep track of the three most recent points
    # We'll maintain a list of (x, f(x)) with at most 3 entries
    # Order: oldest to newest
    
    points = [(x, fx), (x1, f1), (x2, f2)]
    
    # Check if any of the three is already a root
    for px, pf in points:
        if abs(pf) <= 1e-12:
            return px
    
    # Main iteration loop
    max_iter = 200
    for _ in range(max_iter):
        # Select the best candidate from the three points
        # We'll use IQI to predict the next root
        x0_, f0_ = points[0]
        x1_, f1_ = points[1]
        x2_, f2_ = points[2]
        
        # Inverse quadratic interpolation:
        # We fit a quadratic in x as a function of f:
        # x = a*f^2 + b*f + c
        # Using Lagrange interpolation in the inverse sense.
        
        # The formula for IQI step:
        # Let f0, f1, f2 be the function values at x0, x1, x2
        # We want to find f* such that x(f*) = 0
        # x(f) = x0 * (f-f1)(f-f2)/((f0-f1)(f0-f2)) 
        #       + x1 * (f-f0)(f-f2)/((f1-f0)(f1-f2)) 
        #       + x2 * (f-f0)(f-f1)/((f2-f0)(f2-f1))
        # Set x(f) = 0 and solve for f. This gives a quadratic in f.
        
        # Alternatively, use the standard IQI formula:
        # The next x is:
        # x_new = x2 + (f2^2 * (x0 - x1)/((f0 - f2)*(f1 - f2)) 
        #         + f2^2 * (x1 - x2)/((f1 - f2)*(f0 - f2)) ... )
        
        # Let me use the more standard formulation:
        # Given three points (x0, f0), (x1, f1), (x2, f2),
        # the IQI estimate for the root is:
        
        if abs(f0_ - f1_) < 1e-30 or abs(f1_ - f2_) < 1e-30 or abs(f0_ - f2_) < 1e-30:
            # Degenerate, fall back to secant
            if abs(f1_ - f2_) > 1e-30:
                x_new = x2_ - f2_ * (x2_ - x1_) / (f2_ - f1_)
            elif abs(f0_ - f2_) > 1e-30:
                x_new = x2_ - f2_ * (x2_ - x0_) / (f2_ - f0_)
            else:
                # All function values nearly equal, just step
                x_new = x2_ + 0.5 * (x2_ - x1_)
        else:
            # Standard IQI formula
            # x_new = x0 * f0^2 * (x1 - x2) / ((f0 - f1)*(f0 - f2)) 
            #       + x1 * f1^2 * (x2 - x0) / ((f1 - f2)*(f1 - f0)) 
            #       + x2 * f2^2 * (x0 - x1) / ((f2 - f0)*(f2 - f1))
            
            term0 = x0_ * (f0_ ** 2) * (x1_ - x2_) / ((f0_ - f1_) * (f0_ - f2_))
            term1 = x1_ * (f1_ ** 2) * (x2_ - x0_) / ((f1_ - f2_) * (f1_ - f0_))
            term2 = x2_ * (f2_ ** 2) * (x0_ - x1_) / ((f2_ - f0_) * (f2_ - f1_))
            
            x_new = term0 + term1 + term2
        
        # Safeguard: if x_new is not finite or too far from x2_, use secant
        if not math.isfinite(x_new):
            x_new = x2_ - f2_ * (x2_ - x1_) / (f2_ - f1_)
        
        # Limit the step size to avoid wandering too far
        max_step = 10.0 * max(1.0, abs(x2_))
        if abs(x_new - x2_) > max_step:
            # Damp the step
            direction = 1.0 if x_new > x2_ else -1.0
            x_new = x2_ + direction * max_step
        
        # Evaluate
        f_new = f(x_new)
        
        if abs(f_new) <= 1e-12:
            return x_new
        
        # Update points: remove the oldest, add the new
        # Keep the three most recent
        points = points[1:] + [(x_new, f_new)]
        
        # Check convergence
        if abs(f_new) <= 1e-8:
            return x_new
    
    # Return the best point
    best_x = min(points, key=lambda p: abs(p[1]))[0]
    return best_x

# EVOLVE-BLOCK-END
