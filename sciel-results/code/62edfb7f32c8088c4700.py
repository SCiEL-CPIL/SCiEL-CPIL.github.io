import numpy as np

def find_root(f, x0):
    x0 = float(x0)
    fx0 = f(x0)
    if abs(fx0) <= 1e-8:
        return x0

    # Strategy: Secant method with Bisection fallback for robustness.
    # If no sign change is detected, fall back to local minimization of |f(x)|.
    
    # Initial points for Secant
    x1 = x0
    f1 = fx0
    # Choose x2. A small step relative to x0 or absolute.
    # For polynomials, 1.0 is often good. If x0 is large, maybe relative.
    # Let's use 1.0 for simplicity, or 1e-6 if x0 is very small?
    # Actually, if x0 is 0, step 1.0 is fine.
    x2 = x0 + 1.0
    f2 = f(x2)
    
    # Track a bracket if a sign change occurs
    # a, b such that f(a)*f(b) <= 0
    a = None
    b = None
    fa = None
    fb = None
    
    # Keep track of best point (lowest |f(x)|) in case we need to fallback
    best_x = x0
    best_f = fx0
    if abs(f2) < abs(best_f):
        best_x = x2
        best_f = f2

    # Main loop: Secant with Bisection safeguard
    for i in range(50):
        # Check convergence
        if abs(f2) <= 1e-8:
            return x2
        if abs(f1) <= 1e-8:
            return x1
            
        # Check for sign change to set bracket
        if f1 * f2 <= 0:
            a = x1
            b = x2
            fa = f1
            fb = f2
        else:
            # If we have a bracket, and secant step goes outside, use bisection
            # If we don't have a bracket, we just do secant, but monitor divergence
            pass
            
        # Calculate Secant step
        if abs(f2 - f1) < 1e-15:
            # Derivative approx 0, try bisection if bracket exists
            if a is not None and b is not None:
                x_mid = (a + b) / 2.0
                f_mid = f(x_mid)
                # Update bracket
                if fa * f_mid <= 0:
                    b = x_mid
                    fb = f_mid
                else:
                    a = x_mid
                    fa = f_mid
                x1, f1 = x2, f2
                x2, f2 = x_mid, f_mid
                if abs(f2) < abs(best_f):
                    best_x = x2
                    best_f = f2
                continue
            else:
                # No bracket, no derivative. Stuck? Break and minimize.
                break
        
        x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
        
        # Safeguard: If x3 is too far from x2, or if we have a bracket and x3 is outside, use Bisection
        # Define "too far" as more than 2x the current bracket width or 2x step from previous
        # Simple safeguard: If we have a bracket [a,b], restrict x3 to [a,b]
        if a is not None and b is not None:
            # If x3 is outside [a,b], use bisection
            if x3 < min(a, b) or x3 > max(a, b):
                x3 = (a + b) / 2.0
                f3 = f(x3)
                # Update bracket
                if fa * f3 <= 0:
                    b = x3
                    fb = f3
                else:
                    a = x3
                    fa = f3
                x1, f1 = x2, f2
                x2, f2 = x3, f3
                if abs(f2) < abs(best_f):
                    best_x = x2
                    best_f = f2
                continue
            else:
                f3 = f(x3)
                # Update bracket if sign change
                if fa * f3 <= 0:
                    b = x3
                    fb = f3
                else:
                    a = x3
                    fa = f3
                x1, f1 = x2, f2
                x2, f2 = x3, f3
                if abs(f2) < abs(best_f):
                    best_x = x2
                    best_f = f2
                continue
        else:
            # No bracket. Check for divergence.
            # If |x3 - x2| is huge, break to minimize.
            if abs(x3 - x2) > 1e6 * (1 + abs(x0)):
                break
            f3 = f(x3)
            # Check for new bracket
            if f1 * f3 <= 0:
                a = x1
                b = x3
                fa = f1
                fb = f3
            elif f2 * f3 <= 0:
                a = x2
                b = x3
                fa = f2
                fb = f3
            
            x1, f1 = x2, f2
            x2, f2 = x3, f3
            if abs(f2) < abs(best_f):
                best_x = x2
                best_f = f2
            if abs(f1) < abs(best_f):
                best_x = x1
                best_f = f1

    # If loop finishes without finding a root
    if abs(best_f) <= 1e-8:
        return best_x
    
    # Fallback: Local minimization of |f(x)| around best_x
    # Use a small window. Since we don't know the scale, try a few scales.
    # We can use scipy's minimize_scalar on a bounded interval around best_x.
    # This is similar to the current program's fallback, but only triggered if Secant/Bisection fails.
    # To be "new", maybe use a simple grid search + ternary search?
    # Let's use scipy.optimize.minimize_scalar for reliability in the fallback, 
    # but the primary driver is Secant/Bisection.
    
    # Determine bounds for minimization
    # Use the range explored or a default small range
    # If we explored a wide range, use that. Else use best_x +/- 10
    if a is not None and b is not None:
        lo = min(a, b)
        hi = max(a, b)
    else:
        # Heuristic bounds
        scale = max(1.0, abs(best_x))
        lo = best_x - 10 * scale
        hi = best_x + 10 * scale
        
    try:
        from scipy.optimize import minimize_scalar
        # Minimize f(x)^2
        res = minimize_scalar(lambda x: f(x)**2, bounds=(lo, hi), method='bounded')
        if res.success and abs(f(res.x)) <= 1e-8:
            return res.x
        # If minimizer is good enough, return it
        # The task requires |f(x)| <= 1e-8.
        # If minimize_scalar finds a local min that is a root, great.
        # If not, it might fail the check.
        # But for polynomials, the global min of f(x)^2 is likely 0 if a root exists.
        return res.x
    except Exception:
        return best_x
