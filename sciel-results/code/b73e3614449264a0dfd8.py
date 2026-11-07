"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Derivative-free inverse quadratic interpolation (IQI) root finder.
    Fits a quadratic to the last three (x, f(x)) points and inverts it to predict the next x.
    Falls back to bisection or secant if needed for robustness.
    """
    import math

    x = float(x0)
    
    # Evaluate first point
    fx0 = f(x)
    if abs(fx0) <= 1e-8:
        return x
    
    # Generate second point with adaptive step
    h = 1e-2 * max(1.0, abs(x))
    x1 = x + h
    fx1 = f(x1)
    if abs(fx1) <= 1e-8:
        return x1
    
    # If sign change, we have a bracket - use bisection/IQI hybrid
    # If no sign change, try other side
    if fx0 * fx1 > 0:
        x1 = x - h
        fx1 = f(x1)
        if abs(fx1) <= 1e-8:
            return x1
    
    # Now we have two points. Check if bracket exists
    a, fa = x, fx0
    b, fb = x1, fx1
    
    # If no bracket after two tries, expand
    if fa * fb > 0:
        # Try expanding in the direction where |f| is smaller
        if abs(fa) < abs(fb):
            b, fb = a - h, f(a - h)
        else:
            a, fa = b + h, f(b + h)
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
    
    # We now have two points. Generate third for IQI
    # Use the point that's more likely to help: if bracket, midpoint; else extrapolate
    if fa * fb < 0:
        # Have bracket, use midpoint as third point
        x_mid = (a + b) / 2.0
        fx_mid = f(x_mid)
        if abs(fx_mid) <= 1e-8:
            return x_mid
        # Keep bracket endpoints and midpoint
        points = [(a, fa), (x_mid, fx_mid), (b, fb)]
    else:
        # No bracket, extrapolate from the two points
        # Use secant to predict next point
        if abs(fb) < abs(fa):
            x_new = b - fb * (b - a) / (fb - fa)
        else:
            x_new = a - fa * (b - a) / (fb - fa)
        fx_new = f(x_new)
        if abs(fx_new) <= 1e-8:
            return x_new
        # Keep the two closest to zero and the new one
        pts = [(a, fa), (b, fb), (x_new, fx_new)]
        pts.sort(key=lambda p: abs(p[1]))
        points = [pts[0], pts[1], pts[2]]
    
    max_iter = 50
    tol = 1e-8
    
    for _ in range(max_iter):
        # Check all points for root
        for px, pfx in points:
            if abs(pfx) <= tol:
                return px
        
        # Sort points by x for consistent ordering
        points.sort(key=lambda p: p[0])
        x_a, f_a = points[0]
        x_b, f_b = points[1]
        x_c, f_c = points[2]
        
        # Check for distinct x values
        if abs(x_a - x_b) < 1e-14 or abs(x_b - x_c) < 1e-14 or abs(x_a - x_c) < 1e-14:
            # Perturb
            h = 1e-4 * max(1.0, abs(x_c))
            x_pert = x_c + h
            f_pert = f(x_pert)
            points[2] = (x_pert, f_pert)
            continue
        
        # Check for bracket
        has_bracket = False
        bracket_lo = bracket_hi = None
        if f_a * f_b <= 0:
            has_bracket = True
            bracket_lo, bracket_hi = x_a, x_b
        elif f_b * f_c <= 0:
            has_bracket = True
            bracket_lo, bracket_hi = x_b, x_c
        elif f_a * f_c <= 0:
            has_bracket = True
            bracket_lo, bracket_hi = x_a, x_c
        
        # Try IQI
        x_new = None
        d_a = f_a - f_c
        d_b = f_b - f_c
        
        if abs(d_a) > 1e-14 and abs(d_b) > 1e-14:
            M1 = d_a * (f_a + f_c)
            M2 = d_b * (f_b + f_c)
            R1 = x_a - x_c
            R2 = x_b - x_c
            
            det = M1 * d_b - M2 * d_a
            
            if abs(det) > 1e-14:
                A = (R1 * d_b - R2 * d_a) / det
                B = (M1 * R2 - M2 * R1) / det
                C = x_c - A * f_c * f_c - B * f_c
                x_new = C
        
        # If IQI failed or x_new is not finite, try secant
        if x_new is None or not math.isfinite(x_new):
            # Use the two points with smallest |f|
            if abs(f_a) < abs(f_c):
                if abs(f_b) < abs(f_c):
                    # Use a and b
                    if abs(f_b - f_a) > 1e-14:
                        x_new = b - f_b * (b - a) / (f_b - f_a)
                    else:
                        x_new = (a + b) / 2.0
                else:
                    # Use a and c
                    if abs(f_c - f_a) > 1e-14:
                        x_new = c - f_c * (c - a) / (f_c - f_a)
                    else:
                        x_new = (a + c) / 2.0
            else:
                if abs(f_b) < abs(f_a):
                    # Use b and c
                    if abs(f_c - f_b) > 1e-14:
                        x_new = c - f_c * (c - b) / (f_c - f_b)
                    else:
                        x_new = (b + c) / 2.0
                else:
                    # Use a and c
                    if abs(f_c - f_a) > 1e-14:
                        x_new = c - f_c * (c - a) / (f_c - f_a)
                    else:
                        x_new = (a + c) / 2.0
        
        # If we have a bracket, ensure x_new is within bracket
        if has_bracket:
            if x_new < bracket_lo or x_new > bracket_hi:
                # Use bisection
                x_new = (bracket_lo + bracket_hi) / 2.0
        
        # Evaluate
        f_new = f(x_new)
        
        if abs(f_new) <= tol:
            return x_new
        
        # Update points: keep the 3 that best constrain the root
        new_points = [(x_a, f_a), (x_b, f_b), (x_c, f_c), (x_new, f_new)]
        
        # Strategy: if we have a bracket, keep both bracket endpoints and the newest interior point
        # Otherwise, keep the 3 points with smallest |f|
        if has_bracket:
            # Find which pair forms the bracket among new points
            best_bracket = None
            best_width = float('inf')
            for i in range(4):
                for j in range(i+1, 4):
                    pi, pj = new_points[i], new_points[j]
                    if pi[1] * pj[1] <= 0:
                        width = abs(pi[0] - pj[0])
                        if width < best_width:
                            best_width = width
                            best_bracket = (i, j)
            
            if best_bracket:
                i, j = best_bracket
                # Keep the bracket and the point with smallest |f| among the other two
                other_indices = [k for k in range(4) if k != i and k != j]
                other_idx = min(other_indices, key=lambda k: abs(new_points[k][1]))
                points = [new_points[i], new_points[other_idx], new_points[j]]
            else:
                # No bracket found, keep 3 with smallest |f|
                new_points.sort(key=lambda p: abs(p[1]))
                points = new_points[:3]
        else:
            # No bracket, keep 3 with smallest |f|
            new_points.sort(key=lambda p: abs(p[1]))
            points = new_points[:3]
    
    # Final check
    for px, pfx in points:
        if abs(pfx) <= tol:
            return px
    
    return min(points, key=lambda p: abs(p[1]))[0]
# EVOLVE-BLOCK-END
