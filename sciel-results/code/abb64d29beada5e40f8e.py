import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x

    # Phase 1: Find a bracket [a, b]
    a = x
    fa = fx
    
    # Try small steps to find a sign change
    h = 1e-4
    found_bracket = False
    b = None
    fb = None
    
    for _ in range(50):
        if time.time() > deadline:
            return a if abs(fa) < 1e-8 else x
        
        # Try x + h
        x_try = x + h
        fx_try = f(x_try)
        if abs(fx_try) <= 1e-8:
            return x_try
        if fa * fx_try < 0:
            b = x_try
            fb = fx_try
            found_bracket = True
            break
        
        # Try x - h
        x_try2 = x - h
        fx_try2 = f(x_try2)
        if abs(fx_try2) <= 1e-8:
            return x_try2
        if fa * fx_try2 < 0:
            b = x_try2
            fb = fx_try2
            found_bracket = True
            break
            
        # Expand step
        h *= 2.0
        if h > 1e6:
            break

    if not found_bracket:
        # Fallback: Use Newton's method or Secant if no bracket found
        # Try Secant with x0 and x0+h
        x1 = x + 1.0
        fx1 = f(x1)
        for _ in range(50):
            if time.time() > deadline:
                return x
            if abs(fx) <= 1e-8:
                return x
            if fx1 == fx:
                break
            x_new = x - fx * (x1 - x) / (fx1 - fx)
            if x_new == x or x_new == x1:
                x_new = x + 1e-6
            x, x1 = x1, x_new
            fx, fx1 = fx1, f(x1)
            if abs(fx1) < abs(fx):
                fx = fx1
                x = x1
        return x

    # Phase 2: IQI-Brent Hybrid
    # We have bracket [a, b] with fa * fb < 0
    # Points for IQI: (fa, a), (fb, b), and we need a third.
    # Let's start with midpoint or use secant first to get a 3rd point.
    
    # Initial 3 points: a, b, and maybe midpoint? Or just start with 2 and add more.
    # IQI needs 3 points. Let's use a, b, and a point close to the root estimate.
    # Secant estimate: x_s = a - fa * (b - a) / (fb - fa)
    x_s = a - fa * (b - a) / (fb - fa)
    
    # Ensure x_s is within [a, b]
    if x_s <= a or x_s >= b:
        x_s = (a + b) / 2.0
        
    fs = f(x_s)
    if abs(fs) <= 1e-8:
        return x_s
        
    # Update bracket if x_s is better
    if fa * fs < 0:
        b = x_s
        fb = fs
    else:
        a = x_s
        fa = fs

    # Now we have 3 points: (a, fa), (b, fb), and we lost the 3rd one?
    # Let's maintain a list of last 3 points (x, f)
    # Actually, standard Brent's method maintains c (previous b or a).
    # For IQI, we need 3 distinct x values.
    
    # Let's store points in a list, max 3
    pts = [(a, fa), (b, fb)]
    # We need a 3rd point. Let's use the midpoint of the original bracket or the secant point.
    # Let's just use the secant point we calculated, even if it moved the bracket.
    # Wait, if x_s moved the bracket, we have new a, b. We need a 3rd point.
    # Let's use the midpoint of the *current* bracket as the 3rd point for initialization?
    # Or better, just start with 2 points and use Secant until we have 3, then switch to IQI.
    
    x_prev = a
    fx_prev = fa
    x_curr = b
    fx_curr = fb
    
    # We have 2 points. Let's get a 3rd via Secant or Bisection to initialize IQI.
    # Secant:
    if fx_curr != fx_prev:
        x_3rd = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
    else:
        x_3rd = (x_curr + x_prev) / 2.0
        
    # Clamp x_3rd to bracket
    if x_3rd < min(x_prev, x_curr) or x_3rd > max(x_prev, x_curr):
        x_3rd = (x_prev + x_curr) / 2.0
        
    f_3rd = f(x_3rd)
    if abs(f_3rd) <= 1e-8:
        return x_3rd
        
    # Update bracket with x_3rd
    if fx_prev * f_3rd < 0:
        x_curr = x_3rd
        fx_curr = f_3rd
    else:
        x_prev = x_3rd
        fx_prev = f_3rd
        
    # Now we have 3 points: x_prev, x_curr, and... we need to keep the 3rd one distinct.
    # Let's maintain a list of the last 3 points (x, f) that are distinct.
    # Points: p1, p2, p3
    p1 = (x_prev, fx_prev)
    p2 = (x_curr, fx_curr)
    p3 = (x_3rd, f_3rd)
    
    # Sort by x? IQI doesn't require sorting, but distinct x is required.
    # Check distinctness
    if abs(p1[0] - p2[0]) < 1e-12 or abs(p1[0] - p3[0]) < 1e-12 or abs(p2[0] - p3[0]) < 1e-12:
        # Fallback to simple bisection if points are too close
        for _ in range(100):
            if time.time() > deadline:
                return x_curr if abs(fx_curr) < abs(fx_prev) else x_prev
            x_mid = (x_prev + x_curr) / 2.0
            f_mid = f(x_mid)
            if abs(f_mid) <= 1e-8:
                return x_mid
            if fx_prev * f_mid < 0:
                x_curr = x_mid
                fx_curr = f_mid
            else:
                x_prev = x_mid
                fx_prev = f_mid
        return x_curr if abs(fx_curr) < abs(fx_prev) else x_prev

    # IQI Iteration
    for _ in range(100):
        if time.time() > deadline:
            # Return best of the 3 points
            best_x = p1[0]
            best_f = abs(p1[1])
            if abs(p2[1]) < best_f:
                best_x = p2[0]
                best_f = abs(p2[1])
            if abs(p3[1]) < best_f:
                best_x = p3[0]
            return best_x

        # Identify current bracket [a, b] from the 3 points
        # The bracket is defined by the two points with opposite signs.
        # There should be exactly one such pair if the function is continuous and we haven't messed up.
        bracket_a = None
        bracket_b = None
        fa_bracket = None
        fb_bracket = None
        
        # Check pairs
        if p1[1] * p2[1] < 0:
            bracket_a, fa_bracket = p1
            bracket_b, fb_bracket = p2
            p_other = p3
        elif p1[1] * p3[1] < 0:
            bracket_a, fa_bracket = p1
            bracket_b, fb_bracket = p3
            p_other = p2
        elif p2[1] * p3[1] < 0:
            bracket_a, fa_bracket = p2
            bracket_b, fb_bracket = p3
            p_other = p1
        else:
            # No bracket? Should not happen if we update correctly.
            # Fallback to bisection on the widest range
            break

        # Ensure a < b
        if bracket_a[0] > bracket_b[0]:
            bracket_a, bracket_b = bracket_b, bracket_a
            fa_bracket, fb_bracket = fb_bracket, fa_bracket

        a_val = bracket_a[0]
        b_val = bracket_b[0]
        
        # Fit IQI: x = A*f^2 + B*f + C
        # Points: (fa, xa), (fb, xb), (fo, xo)
        fa, xa = bracket_a
        fb, xb = bracket_b
        fo, xo = p_other
        
        # Solve for A, B, C
        # xa = A*fa^2 + B*fa + C
        # xb = A*fb^2 + B*fb + C
        # xo = A*fo^2 + B*fo + C
        
        # C is the value at f=0, which is the root we want.
        # Using Lagrange interpolation for x as a function of f:
        # x(f) = xa * (f-fb)(f-fo) / ((fa-fb)(fa-fo)) + 
        #       xb * (f-fa)(f-fo) / ((fb-fa)(fb-fo)) + 
        #       xo * (f-fa)(f-fb) / ((fo-fa)(fo-fb))
        
        # We want x(0).
        try:
            denom1 = (fa - fb) * (fa - fo)
            denom2 = (fb - fa) * (fb - fo)
            denom3 = (fo - fa) * (fo - fb)
            
            if abs(denom1) < 1e-15 or abs(denom2) < 1e-15 or abs(denom3) < 1e-15:
                # Degenerate case, use bisection
                x_new = (a_val + b_val) / 2.0
            else:
                term1 = xa * (-fb) * (-fo) / denom1
                term2 = xb * (-fa) * (-fo) / denom2
                term3 = xo * (-fa) * (-fb) / denom3
                x_new = term1 + term2 + term3
        except ZeroDivisionError:
            x_new = (a_val + b_val) / 2.0

        # Safety: Ensure x_new is within [a_val, b_val]
        if x_new <= a_val or x_new >= b_val:
            x_new = (a_val + b_val) / 2.0
            
        f_new = f(x_new)
        if abs(f_new) <= 1e-8:
            return x_new

        # Update points
        # The new point replaces the one that was NOT in the bracket?
        # Or we just keep the 3 points that define the current state?
        # Standard approach: The new point becomes one of the bracket endpoints.
        # The point that is no longer in the bracket (p_other) is replaced by the new point?
        # Actually, we should keep the 3 most recent points or the ones defining the bracket.
        # Let's replace p_other with the new point, as p_other is the one outside the current bracket.
        # Wait, if we replace p_other, we lose the 3rd point for IQI?
        # No, the new bracket is formed by (bracket_a, new) or (new, bracket_b).
        # The 3 points for the NEXT iteration should be:
        # 1. The new point (x_new, f_new)
        # 2. The two points of the current bracket (bracket_a, bracket_b)
        # This maintains 3 points, two of which form the bracket.
        
        p1 = bracket_a
        p2 = bracket_b
        p3 = (x_new, f_new)
        
        # Check if we are stuck (no progress)
        if abs(b_val - a_val) < 1e-12:
            break

    # Return best of last 3 points
    best_x = p1[0]
    best_f = abs(p1[1])
    if abs(p2[1]) < best_f:
        best_x = p2[0]
        best_f = abs(p2[1])
    if abs(p3[1]) < best_f:
        best_x = p3[0]
        
    return best_x
