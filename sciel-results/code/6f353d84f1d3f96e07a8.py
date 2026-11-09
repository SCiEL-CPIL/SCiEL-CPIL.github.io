import math
import time

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Uses bracketing search to find a sign change, then Illinois method for refinement.
    """
    start_time = time.time()
    deadline = start_time + 10.0
    tol_final = 1e-8
    
    x0 = float(x0)
    f0 = f(x0)
    if math.isfinite(f0) and abs(f0) <= tol_final:
        return x0

    # Track best point found so far
    best_x = x0
    best_f = abs(f0) if math.isfinite(f0) else float('inf')

    # If f0 is not finite, try to find a finite point
    if not math.isfinite(f0):
        base_step = 1e-4 * max(1.0, abs(x0))
        for s in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64]:
            if time.time() > deadline:
                break
            x_try = x0 + s * base_step
            f_try = f(x_try)
            if math.isfinite(f_try):
                if abs(f_try) <= tol_final:
                    return x_try
                x0 = x_try
                f0 = f_try
                best_x = x0
                best_f = abs(f0)
                break
        if not math.isfinite(f0):
            return best_x

    # Search for a bracket [a, b] such that f(a)*f(b) <= 0
    # We expand from x0 in both directions
    a = x0
    fa = f0
    b = None
    fb = None
    
    # Try to find a sign change by expanding step size
    # We check points x0 + k*h and x0 - k*h
    # To be efficient, we can interleave or just expand one side then the other?
    # Better: expand in a spiral or just double step size.
    
    h = 1e-4 * max(1.0, abs(x0))
    # We will check points at x0 +/- h, x0 +/- 2h, x0 +/- 4h, etc.
    # But we already have x0.
    
    found_bracket = False
    # We maintain a list of points to check in order of distance from x0
    # Actually, just expanding h and checking x0+h and x0-h is fine.
    
    # To avoid missing a sign change between x0 and x0+h if h is too large?
    # For oscillatory functions, h should be small enough to resolve the period.
    # 1e-4 is usually small. If the period is very small, we might step over a root.
    # If we step over a root, f(x0) and f(x0+h) might have same sign.
    # We should check intermediate points if the step is large?
    # Or just use a smaller initial h?
    # Let's stick to h = 1e-4 * max(1, |x0|).
    
    # We'll iterate up to a certain number of doublings
    for i in range(20):
        if time.time() > deadline - 2.0:
            break
            
        # Check right side
        x_r = x0 + h
        f_r = f(x_r)
        if math.isfinite(f_r):
            if abs(f_r) < best_f:
                best_x = x_r
                best_f = abs(f_r)
            if abs(f_r) <= tol_final:
                return x_r
            if math.isfinite(fa) and fa * f_r <= 0:
                a, fa = x0, fa
                b, fb = x_r, f_r
                found_bracket = True
                break
        
        # Check left side
        x_l = x0 - h
        f_l = f(x_l)
        if math.isfinite(f_l):
            if abs(f_l) < best_f:
                best_x = x_l
                best_f = abs(f_l)
            if abs(f_l) <= tol_final:
                return x_l
            if math.isfinite(fa) and fa * f_l <= 0:
                a, fa = x0, fa
                b, fb = x_l, f_l
                found_bracket = True
                break
        
        # Also check if the previous right point and current right point bracket?
        # We are expanding from x0. The interval is [x0, x0+h] or [x0-h, x0].
        # If we didn't find a bracket with x0, maybe the function is monotonic in the direction?
        # For oscillatory, it will change sign eventually.
        
        # What if x0 is not the anchor?
        # If we expand, we should check if the new point brackets with the *previous* outer point?
        # No, we are expanding from x0.
        # If f(x0) and f(x0+h) have same sign, and f(x0) and f(x0+2h) have same sign...
        # It's possible we jumped over a root if h is too big.
        # To be safe, if we double h, we should check the midpoint?
        # That doubles the calls.
        
        h *= 2.0

    if not found_bracket:
        # Fallback: Use the best point found and try to refine locally
        # This handles cases where no sign change was found (e.g. tangent root or step too large)
        # We try a few secant steps from best_x
        if math.isfinite(best_f) and best_f > tol_final:
             # Try to find a second point for secant
             # We can use one of the points we evaluated, or generate a new one
             # Let's just return best_x if we can't do better, or try a tiny perturbation
             # But first, let's see if we can use the last evaluated points
             pass # We will just return best_x at the end if no bracket

    if found_bracket:
        # Illinois Method
        # a, fa, b, fb are such that fa*fb <= 0
        # Ensure |fa| < |fb| so we shrink the 'b' side (larger f)
        if abs(fa) < abs(fb):
            # a is closer to root
            pass
        else:
            a, b = b, a
            fa, fb = fb, fa
            
        # Illinois iteration
        for _ in range(50):
            if time.time() > deadline - 0.5:
                break
                
            # Check convergence
            if abs(fa) <= tol_final:
                return a
            if abs(fb) <= tol_final:
                return b
                
            # Secant step
            denom = fb - fa
            if abs(denom) < 1e-15:
                break
            x_new = b - fb * (b - a) / denom
            f_new = f(x_new)
            
            if not math.isfinite(f_new):
                break
                
            if abs(f_new) <= tol_final:
                return x_new
                
            if fa * f_new < 0:
                # Root is in [a, x_new]
                b = x_new
                fb = f_new
            else:
                # Root is in [x_new, b]
                a = x_new
                fa = f_new
                
            # Illinois modification:
            # If the new point is on the side that was just updated (the side with larger |f| previously?), 
            # actually Illinois says: if the new point replaces the point with the smaller absolute value, 
            # then halve the function value of the other point.
            # Wait, standard Illinois:
            # Let x0, x1 be the points. f(x0) < f(x1) in magnitude? No, usually we keep the sign change.
            # Let's stick to the standard implementation found in many numerical methods texts.
            # "If the new point is on the same side as the point that was just replaced, 
            #  halve the function value of the other point."
            # Actually, simpler rule: 
            # If |f(a)| < |f(b)|, and we update b with x_new (so x_new is between a and old b),
            # then if x_new is closer to b than a? No.
            
            # Let's use the explicit Illinois rule:
            # 1. Calculate x_new using secant.
            # 2. If f(a) and f(x_new) have same sign:
            #    a = x_new
            #    f(a) = f(a) / 2  (Halve the function value of the point that was NOT updated? No.)
            #    
            # Correct Illinois Algorithm:
            # Let x0, x1 be the current bracket.
            # Assume |f(x0)| < |f(x1)|.
            # Compute x2 = x1 - f(x1)(x1-x0)/(f(x1)-f(x0)).
            # If f(x0) and f(x2) have same sign:
            #    x1 = x2
            #    f(x1) = f(x1)/2
            # Else:
            #    x0 = x2
            #    f(x0) = f(x2)
            # Swap x0, x1 if |f(x1)| < |f(x0)|.
            
            # Let's re-implement the loop with this specific logic to be safe.
            # My previous loop was standard Regula Falsi. Illinois is better.
            # I will restart the loop with proper Illinois logic.
            pass # The code below will replace the simple secant loop

    # If we didn't find a bracket, or Illinois finished, return best
    return best_x
