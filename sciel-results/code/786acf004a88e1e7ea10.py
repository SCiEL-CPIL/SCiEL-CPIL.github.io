"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Find a root of f starting near x0 using polynomial interpolation (inverse quadratic / cubic).
    Strategy:
      1. Start with x0, evaluate f.
      2. If not close enough, take a small step to get a second point.
      3. Take another step to get a third point.
      4. Fit a quadratic (or cubic if 4 points) to (x, f(x)) pairs and find the root of the interpolant.
      5. Use that root as the next candidate. If it's not good, refine with more points or fall back to bisection-like logic.
      6. Repeat until |f(x)| <= 1e-10 or max iterations.
    
    To minimize calls, we aim for convergence in 3-6 evaluations.
    """
    x = float(x0)
    max_iter = 30
    tol = 1e-10  # tighter than 1e-8 to be safe
    
    # Store points: list of (x_val, f_val)
    points = []
    
    def add_point(xv):
        fv = f(xv)
        if not math.isfinite(fv):
            return None
        points.append((xv, fv))
        return fv
    
    # Initial evaluation
    f0 = add_point(x)
    if f0 is None:
        return x
    if abs(f0) <= tol:
        return x
    
    # Take a small step to get second point
    # Adaptive step size
    scale = max(1.0, abs(x))
    h = 1e-4 * scale
    x1 = x + h
    f1 = add_point(x1)
    if f1 is None:
        return x
    if abs(f1) <= tol:
        return x1
    
    # If signs differ, we have a bracket. Use bisection + interpolation.
    # If same sign, take another step in opposite direction or larger step.
    if f0 * f1 < 0:
        # We have a bracket [x, x1]
        lo, hi = (x, x1) if f0 < 0 else (x1, x)
        f_lo, f_hi = (f0, f1) if f0 < 0 else (f1, f0)
        
        # Inverse quadratic interpolation using 3 points would be ideal, but we only have 2.
        # Use secant method for next step, then switch to interpolation.
        # Secant:
        if f1 != f0:
            x2 = x1 - f1 * (x1 - x) / (f1 - f0)
        else:
            x2 = (x + x1) / 2.0
        
        # Keep x2 within bracket
        if not (lo < x2 < hi):
            x2 = (lo + hi) / 2.0
        
        f2 = add_point(x2)
        if f2 is None:
            return x2
        if abs(f2) <= tol:
            return x2
        
        # Now we have 3 points. Use inverse quadratic interpolation.
        # But we need to be careful with bracketing.
        # Let's just iterate with a robust method: keep bracket, use interpolation if possible, else bisection.
        
        for _ in range(max_iter - 3):
            if abs(f2) <= tol:
                return x2
            
            # Update bracket
            if f2 * f_lo < 0:
                hi, f_hi = x2, f2
            else:
                lo, f_lo = x2, f2
            
            # Try inverse quadratic interpolation using the 3 most recent points
            # Points are in points list. Use last 3.
            if len(points) >= 3:
                p1 = points[-3]
                p2 = points[-2]
                p3 = points[-1]
                
                # Inverse quadratic interpolation: fit quadratic through (f1, x1), (f2, x2), (f3, x3)
                # and find x where f=0.
                # Using Lagrange interpolation for x as function of f.
                f_a, x_a = p1[1], p1[0]
                f_b, x_b = p2[1], p2[0]
                f_c, x_c = p3[1], p3[0]
                
                # Check for duplicates or near-duplicates in f
                denom_a = (f_a - f_b) * (f_a - f_c)
                denom_b = (f_b - f_a) * (f_b - f_c)
                denom_c = (f_c - f_a) * (f_c - f_b)
                
                if abs(denom_a) > 1e-30 and abs(denom_b) > 1e-30 and abs(denom_c) > 1e-30:
                    x_next = (x_a * denom_a + x_b * denom_b + x_c * denom_c) / (denom_a + denom_b + denom_c)
                    # Actually, the standard formula for inverse quadratic is:
                    # x_new = (x_a * f_b * f_c / ((f_a-f_b)*(f_a-f_c)) + x_b * f_a * f_c / ((f_b-f_a)*(f_b-f_c)) + x_c * f_a * f_b / ((f_c-f_a)*(f_c-f_b))) / (f_b*f_c/((f_a-f_b)*(f_a-f_c)) + f_a*f_c/((f_b-f_a)*(f_b-f_c)) + f_a*f_b/((f_c-f_a)*(f_c-f_b)))
                    # Let me use the simpler Lagrange form for x(f) at f=0:
                    # L_a(0) = ((0-f_b)(0-f_c)) / ((f_a-f_b)(f_a-f_c))
                    # L_b(0) = ((0-f_a)(0-f_c)) / ((f_b-f_a)(f_b-f_c))
                    # L_c(0) = ((0-f_a)(0-f_b)) / ((f_c-f_a)(f_c-f_b))
                    # x_new = x_a * L_a(0) + x_b * L_b(0) + x_c * L_c(0)
                    
                    La = (f_b * f_c) / (denom_a)
                    Lb = (f_a * f_c) / (denom_b)
                    Lc = (f_a * f_b) / (denom_c)
                    
                    Lsum = La + Lb + Lc
                    if abs(Lsum) > 1e-30:
                        x_new = (x_a * La + x_b * Lb + x_c * Lc) / Lsum
                    else:
                        x_new = (lo + hi) / 2.0
                else:
                    x_new = (lo + hi) / 2.0
                
                # Ensure x_new is in bracket
                if not (lo < x_new < hi):
                    x_new = (lo + hi) / 2.0
                
            else:
                # Bisection
                x_new = (lo + hi) / 2.0
            
            f_new = add_point(x_new)
            if f_new is None:
                return x_new
            if abs(f_new) <= tol:
                return x_new
            
            x2 = x_new
            f2 = f_new
            
        return x2
    
    else:
        # No bracket yet. Take a step in the opposite direction or a larger step.
        x2 = x - h
        f2 = add_point(x2)
        if f2 is None:
            return x
        if abs(f2) <= tol:
            return x2
        
        # Now we have 3 points: (x, f0), (x1, f1), (x2, f2)
        # Try to find a bracket among these or use interpolation.
        
        # Check all pairs for sign changes
        bracket = None
        for i in range(3):
            for j in range(i+1, 3):
                if points[i][1] * points[j][1] < 0:
                    bracket = (points[i], points[j])
                    break
            if bracket:
                break
        
        if bracket:
            lo_pt, hi_pt = bracket
            lo, hi = (lo_pt[0], hi_pt[0]) if lo_pt[1] < 0 else (hi_pt[0], lo_pt[0])
            f_lo, f_hi = (lo_pt[1], hi_pt[1]) if lo_pt[1] < 0 else (hi_pt[1], lo_pt[1])
            
            # Use the 3 points for inverse quadratic
            p1, p2, p3 = points[0], points[1], points[2]
            f_a, x_a = p1[1], p1[0]
            f_b, x_b = p2[1], p2[0]
            f_c, x_c = p3[1], p3[0]
            
            denom_a = (f_a - f_b) * (f_a - f_c)
            denom_b = (f_b - f_a) * (f_b - f_c)
            denom_c = (f_c - f_a) * (f_c - f_b)
            
            if abs(denom_a) > 1e-30 and abs(denom_b) > 1e-30 and abs(denom_c) > 1e-30:
                La = (f_b * f_c) / denom_a
                Lb = (f_a * f_c) / denom_b
                Lc = (f_a * f_b) / denom_c
                Lsum = La + Lb + Lc
                if abs(Lsum) > 1e-30:
                    x_new = (x_a * La + x_b * Lb + x_c * Lc) / Lsum
                else:
                    x_new = (lo + hi) / 2.0
            else:
                x_new = (lo + hi) / 2.0
            
            if not (lo < x_new < hi):
                x_new = (lo + hi) / 2.0
            
            f_new = add_point(x_new)
            if f_new is None:
                return x_new
            if abs(f_new) <= tol:
                return x_new
            
            # Continue with bracketing iteration
            for _ in range(max_iter - 4):
                if abs(f_new) <= tol:
                    return x_new
                
                if f_new * f_lo < 0:
                    hi, f_hi = x_new, f_new
                else:
                    lo, f_lo = x_new, f_new
                
                # Use last 3 points for inverse quadratic
                if len(points) >= 3:
                    p1 = points[-3]
                    p2 = points[-2]
                    p3 = points[-1]
                    f_a, x_a = p1[1], p1[0]
                    f_b, x_b = p2[1], p2[0]
                    f_c, x_c = p3[1], p3[0]
                    
                    denom_a = (f_a - f_b) * (f_a - f_c)
                    denom_b = (f_b - f_a) * (f_b - f_c)
                    denom_c = (f_c - f_a) * (f_c - f_b)
                    
                    if abs(denom_a) > 1e-30 and abs(denom_b) > 1e-30 and abs(denom_c) > 1e-30:
                        La = (f_b * f_c) / denom_a
                        Lb = (f_a * f_c) / denom_b
                        Lc = (f_a * f_b) / denom_c
                        Lsum = La + Lb + Lc
                        if abs(Lsum) > 1e-30:
                            x_next = (x_a * La + x_b * Lb + x_c * Lc) / Lsum
                        else:
                            x_next = (lo + hi) / 2.0
                    else:
                        x_next = (lo + hi) / 2.0
                    
                    if not (lo < x_next < hi):
                        x_next = (lo + hi) / 2.0
                else:
                    x_next = (lo + hi) / 2.0
                
                f_next = add_point(x_next)
                if f_next is None:
                    return x_next
                if abs(f_next) <= tol:
                    return x_next
                
                x_new = x_next
                f_new = f_next
            
            return x_new
        else:
            # No bracket found with 3 points. Try a larger step or use polynomial root finding.
            # Fit a quadratic to the 3 points and find its roots.
            p1, p2, p3 = points[0], points[1], points[2]
            x_a, f_a = p1
            x_b, f_b = p2
            x_c, f_c = p3
            
            # Quadratic through (x_a, f_a), (x_b, f_b), (x_c, f_c)
            # Using Lagrange form: q(x) = f_a * L_a(x) + f_b * L_b(x) + f_c * L_c(x)
            # Find roots of q(x) = 0.
            # This is a quadratic in x. Let's compute coefficients.
            # q(x) = A*x^2 + B*x + C
            # Using the standard formula for quadratic interpolation coefficients.
            
            # Alternatively, just try a few more steps with larger h.
            h2 = 1e-2 * scale
            x3 = x + h2
            f3 = add_point(x3)
            if f3 is None:
                return x
            if abs(f3) <= tol:
                return x3
            
            # Check for bracket with new point
            for i in range(len(points)-1):
                for j in range(i+1, len(points)):
                    if points[i][1] * points[j][1] < 0:
                        lo_pt, hi_pt = points[i], points[j]
                        lo, hi = (lo_pt[0], hi_pt[0]) if lo_pt[1] < 0 else (hi_pt[0], lo_pt[0])
                        f_lo, f_hi = (lo_pt[1], hi_pt[1]) if lo_pt[1] < 0 else (hi_pt[1], lo_pt[1])
                        
                        # Bisect to converge
                        for _ in range(max_iter - 5):
                            x_mid = (lo + hi) / 2.0
                            f_mid = add_point(x_mid)
                            if f_mid is None:
                                return x_mid
                            if abs(f_mid) <= tol:
                                return x_mid
                            if f_mid * f_lo < 0:
                                hi, f_hi = x_mid, f_mid
                            else:
                                lo, f_lo = x_mid, f_mid
                        return (lo + hi) / 2.0
            
            # If still no bracket, return the point with smallest |f|
            best_x = min(points, key=lambda p: abs(p[1]))[0]
            return best_x
    
# EVOLVE-BLOCK-END
