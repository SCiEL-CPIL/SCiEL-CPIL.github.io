import math

def find_root(f, x0):
    tol = 1e-8
    x = float(x0)
    
    # Initial evaluation
    fx = f(x)
    calls = 1
    
    if abs(fx) <= tol:
        return x
    if not math.isfinite(fx):
        # If initial point is bad, try a small step
        x = x + 1.0
        fx = f(x)
        calls += 1
        if abs(fx) <= tol:
            return x
        if not math.isfinite(fx):
            return x # Fail fast if we can't even get a finite value

    # Phase 1: Find a bracket [a, b]
    a = x
    fa = fx
    b = None
    fb = None
    
    # Expand to find a sign change
    # We try expanding in both directions
    step = 1.0
    # Try positive direction first
    for i in range(20):
        x_try = a + step
        f_try = f(x_try)
        calls += 1
        if abs(f_try) <= tol:
            return x_try
        if fa * f_try < 0:
            b = x_try
            fb = f_try
            break
        step *= 2.0
        
    if b is None:
        # Try negative direction
        step = 1.0
        for i in range(20):
            x_try = a - step
            f_try = f(x_try)
            calls += 1
            if abs(f_try) <= tol:
                return x_try
            if fa * f_try < 0:
                b = x_try
                fb = f_try
                break
            step *= 2.0

    if b is None:
        # No bracket found. Return best effort or fail.
        # For oscillatory functions, this might happen if we start far away.
        # We return the point with smallest |f| found so far, which is just a.
        return a

    # Phase 2: Solve using Inverse Quadratic Interpolation (IQUOIN) with Bisection fallback
    # We have a, b. We need a third point c for quadratic interpolation.
    # Let's use the initial point 'x' (which is 'a') and maybe a midpoint or another point?
    # Actually, we have a and b. We need a third distinct point.
    # We can use the point 'x' if it's distinct from a and b. But x == a.
    # So we need to evaluate a new point to have 3 points.
    # Let's evaluate the midpoint of the current bracket as the third point.
    
    c = (a + b) / 2.0
    fc = f(c)
    calls += 1
    
    if abs(fc) <= tol:
        return c
    
    # Update bracket if c is better or changes sign
    if fa * fc < 0:
        b = c
        fb = fc
    elif fb * fc < 0:
        a = c
        fa = fc
    else:
        # fc has same sign as fa and fb. 
        # Keep the bracket [a, b] but remember c for interpolation history.
        pass

    # History for IQUOIN: (x, f)
    # We want to fit x = g(f) using quadratic.
    # Points: (fa, a), (fb, b), (fc, c)
    
    # To ensure we always make progress, we will use a loop.
    # We maintain a list of the last 3 points used for interpolation.
    # However, standard IQUOIN requires 3 distinct points.
    
    x1, f1 = a, fa
    x2, f2 = b, fb
    x3, f3 = c, fc
    
    # Ensure distinct x values
    if abs(x1 - x2) < 1e-15 or abs(x1 - x3) < 1e-15 or abs(x2 - x3) < 1e-15:
        # If points are too close, just do bisection
        for i in range(100):
            mid = (x1 + x2) / 2.0
            fm = f(mid)
            calls += 1
            if abs(fm) <= tol:
                return mid
            if f1 * fm < 0:
                x2 = mid
                f2 = fm
            else:
                x1 = mid
                f1 = fm
            if abs(x2 - x1) < 1e-12:
                return x1 if abs(f1) < abs(f2) else x2
        return x1 if abs(f1) < abs(f2) else x2

    # Main loop
    for i in range(100):
        # Check convergence
        if abs(f1) <= tol: return x1
        if abs(f2) <= tol: return x2
        if abs(f3) <= tol: return x3
        
        # Calculate Inverse Quadratic Interpolation step
        # We fit x = A*f^2 + B*f + C
        # Using Lagrange basis for x as function of f:
        # x_new = x1 * L1(f=0) + x2 * L2(f=0) + x3 * L3(f=0)
        # L1(0) = (0 - f2)(0 - f3) / ((f1 - f2)(f1 - f3))
        # L2(0) = (0 - f1)(0 - f3) / ((f2 - f1)(f2 - f3))
        # L3(0) = (0 - f1)(0 - f2) / ((f3 - f1)(f3 - f2))
        
        # Handle division by zero if f values are identical
        if abs(f1 - f2) < 1e-15 or abs(f1 - f3) < 1e-15 or abs(f2 - f3) < 1e-15:
            # Fall back to bisection
            x_mid = (x1 + x2) / 2.0
            f_mid = f(x_mid)
            calls += 1
            if abs(f_mid) <= tol:
                return x_mid
            
            # Update history
            # We replace the point with the largest |f|? Or just shift?
            # Simple strategy: Shift history.
            x1, f1 = x2, f2
            x2, f2 = x3, f3
            x3, f3 = x_mid, f_mid
            
            # Maintain bracket
            if f1 * f_mid < 0:
                x2 = x_mid
                f2 = f_mid
            elif f2 * f_mid < 0: # This logic is tricky if we just shift.
                # Let's restart bracketing logic properly
                pass
            
            # To keep it simple and robust, if IQUOIN fails due to close f values,
            # let's just do a bisection step and update the bracket [a,b] directly.
            
            # Let's restart with a cleaner state management for the loop below.
            # We will manage a bracket [lo, hi] and a list of recent points.
            
            # Let's rewrite the loop structure to be safer.
            break
        
        L1 = (f2 * f3) / ((f1 - f2) * (f1 - f3))
        L2 = (f1 * f3) / ((f2 - f1) * (f2 - f3))
        L3 = (f1 * f2) / ((f3 - f1) * (f3 - f2))
        
        x_new = x1 * L1 + x2 * L2 + x3 * L3
        
        if not math.isfinite(x_new):
            x_new = (x1 + x2) / 2.0
            
        # Check if x_new is within the current bracket [min(x1,x2,x3), max(x1,x2,x3)]?
        # Actually, we should maintain a specific bracket [a, b] where fa*fb < 0.
        # The points x1, x2, x3 are just for interpolation.
        
        # Let's identify the current bracket from the history.
        # This is getting complex. Let's use a simpler, robust variant:
        # Keep a bracket [a, b]. Keep a third point c.
        # Use IQUOIN on a, b, c to predict x_new.
        # If x_new is outside [a, b], use bisection midpoint.
        # Evaluate f(x_new).
        # Update bracket [a, b] based on sign of f(x_new).
        # Update c to be the point among {a, b, x_new} that is not in the new bracket? 
        # Or just keep the last 3 evaluated points.
        
        # Let's restart the loop logic with a cleaner state.
        # State: a, fa, b, fb (bracket), c, fc (extra point)
        
        # We already have a, b, c.
        # Let's map x1,x2,x3 to a,b,c for clarity.
        # We don't know which of x1,x2,x3 is the bracket endpoint.
        
        # Let's discard the previous complex history and use a standard IQUOIN with bracket.
        
        # Re-initialize state from a, b, c
        # Find which pair forms the bracket
        # fa, fb, fc are known.
        
        # Let's just use the points a, b, c.
        # Determine bracket [lo, hi]
        if fa * fb < 0:
            lo, hi = a, b
            flo, fhi = fa, fb
            # c is the extra point
        elif fa * fc < 0:
            lo, hi = a, c
            flo, fhi = fa, fc
            # b is the extra point
        else:
            lo, hi = b, c
            flo, fhi = fb, fc
            # a is the extra point
            
        # Now loop
        for j in range(100):
            if abs(flo) <= tol: return lo
            if abs(fhi) <= tol: return hi
            
            # We need 3 points for IQUOIN. We have lo, hi, and the "extra" point.
            # Let's call the extra point x_e, f_e.
            if fa * fb < 0:
                x_e, f_e = c, fc
            elif fa * fc < 0:
                x_e, f_e = b, fb
            else:
                x_e, f_e = a, fa
                
            # Check if points are distinct enough
            if abs(lo - hi) < 1e-15:
                return lo if abs(flo) < abs(fhi) else hi
            
            # Calculate IQUOIN step
            # Fit x(f) through (flo, lo), (fhi, hi), (f_e, x_e)
            # We want x such that f(x) = 0.
            
            # If f values are too close, use bisection
            if abs(flo - fhi) < 1e-15 or abs(flo - f_e) < 1e-15 or abs(fhi - f_e) < 1e-15:
                x_mid = (lo + hi) / 2.0
                f_mid = f(x_mid)
                calls += 1
                if abs(f_mid) <= tol: return x_mid
                
                # Update bracket
                if flo * f_mid < 0:
                    hi = x_mid
                    fhi = f_mid
                    # Update extra point: keep the one that is not lo/hi?
                    # Simple strategy: replace the extra point with the new midpoint
                    x_e = x_mid
                    f_e = f_mid
                else:
                    lo = x_mid
                    flo = f_mid
                    x_e = x_mid
                    f_e = f_mid
                    
                continue

            # Lagrange coefficients for x at f=0
            # L_lo = (fhi * f_e) / ((flo - fhi) * (flo - f_e))
            # L_hi = (flo * f_e) / ((fhi - flo) * (fhi - f_e))
            # L_e  = (flo * fhi) / ((f_e - flo) * (f_e - fhi))
            
            L_lo = (fhi * f_e) / ((flo - fhi) * (flo - f_e))
            L_hi = (flo * f_e) / ((fhi - flo) * (fhi - f_e))
            L_e  = (flo * fhi) / ((f_e - flo) * (f_e - fhi))
            
            x_new = lo * L_lo + hi * L_hi + x_e * L_e
            
            if not math.isfinite(x_new):
                x_new = (lo + hi) / 2.0
                
            # Check if x_new is within bracket
            if x_new < min(lo, hi) or x_new > max(lo, hi):
                x_new = (lo + hi) / 2.0
                
            f_new = f(x_new)
            calls += 1
            
            if abs(f_new) <= tol:
                return x_new
                
            # Update bracket
            if flo * f_new < 0:
                hi = x_new
                fhi = f_new
            else:
                lo = x_new
                flo = f_new
                
            # Update extra point
            # Heuristic: Keep the point that is furthest from the new bracket midpoint?
            # Or just replace the extra point with the new point if it's "better"?
            # Common strategy: If x_new is closer to the root than x_e, replace x_e with x_new?
            # Actually, for IQUOIN, we just need 3 distinct points.
            # If we replace x_e with x_new, we lose the old x_e.
            # Let's replace the point among {lo, hi, x_e} that is NOT in the new bracket [lo, hi].
            # Wait, lo and hi are in the new bracket.
            # So x_e is the one outside? No, x_e was outside the *previous* bracket.
            # The new bracket is [lo, hi] (one of them changed).
            # The point that is NOT lo and NOT hi is the new x_e.
            
            if abs(x_new - lo) > 1e-15 and abs(x_new - hi) > 1e-15:
                # x_new is the new extra point?
                # No, x_new is inside the new bracket.
                # So the "extra" point is the one of {old_lo, old_hi, old_x_e} that is not the new lo and not the new hi.
                
                # Let's just track the 3 points: p1, p2, p3.
                # This is getting complicated to manage state perfectly.
                # Simpler: Just use Secant Method with Bisection Fallback.
                # Secant uses 2 points. Bisection ensures convergence.
                # Secant is faster than Bisection, IQUOIN is faster than Secant but harder to manage robustly.
                # Given the "oscillatory" nature, Secant can jump over roots.
                # Bisection is safe.
                
                # Let's switch strategy to:
                # 1. Try Secant step.
                # 2. If Secant step is outside bracket, use Bisection.
                # 3. If Secant step doesn't reduce interval by at least 50% (or similar), use Bisection.
                
                # This is a standard robust hybrid (like Illinois or Brent's method but simpler).
                
                # Let's implement this hybrid in the loop.
                
                # We need to break out of the IQUOIN attempt and switch to Secant/Bisection.
                # For the sake of this code generation, let's just use the Secant/Bisection hybrid
                # which is simpler and very robust.
                
                # I will replace the entire Phase 2 with a Secant/Bisection hybrid.
                
                pass # Placeholder to break logic if I switch to Secant

            # To keep the code clean, I will implement the Secant/Bisection hybrid directly
            # instead of the complex IQUOIN state management which is prone to bugs.
            
            # Restarting Phase 2 logic in my head:
            # Use a, b as bracket.
            # Use c as previous point for Secant.
            
            # Let's just write the Secant/Bisection hybrid now.
            
            break # Break out of IQUOIN attempt, switch to Secant/Bisection below.

        # --- Secant/Bisection Hybrid ---
        # We have a bracket [lo, hi] and values flo, fhi.
        # We need a third point for Secant. Let's use x_e, f_e.
        # If x_e is not in [lo, hi], it's fine.
        
        # Actually, let's just use the standard "Brent's method" logic simplified:
        # Maintain a, b, c.
        # a, b are the bracket.
        # c is the previous point.
        
        # Let's re-initialize a, b, c for the hybrid loop.
        # We have fa, fb, fc.
        # Find bracket [a, b].
        if fa * fb < 0:
            a, b = a, b
            fa, fb = fa, fb
            c = c
            fc = fc
        elif fa * fc < 0:
            a, b = a, c
            fa, fb = fa, fc
            c = b
            fc = fb
        else:
            a, b = b, c
            fa, fb = fb, fc
            c = a
            fc = fa
            
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        # c is the other point.
        
        for k in range(100):
            if abs(fa) <= tol: return a
            if abs(fb) <= tol: return b
            
            # Secant step
            if fa != fb:
                x_sec = b - fb * (b - a) / (fb - fa)
            else:
                x_sec = (a + b) / 2.0
                
            # Bisection step
            x_bis = (a + b) / 2.0
            
            # Use Secant if it's within bracket and "reasonable"
            # Reasonable: x_sec is between a and b, and |x_sec - x_bis| is not too small?
            # Actually, standard rule: Use Secant if it's inside bracket.
            # If outside, use Bisection.
            
            if a < x_sec < b:
                x_next = x_sec
            else:
                x_next = x_bis
                
            f_next = f(x_next)
            calls += 1
            
            if abs(f_next) <= tol:
                return x_next
                
            # Update bracket
            if fa * f_next < 0:
                b = x_next
                fb = f_next
            else:
                a = x_next
                fa = f_next
                
            # Update c?
            # For Secant, we just need the last 2 points.
            # But we are using the bracket endpoints for Secant calculation.
            # This is effectively the Secant method constrained to the bracket.
            
            if abs(b - a) < 1e-12:
                return a if abs(fa) < abs(fb) else b

        return a if abs(fa) < abs(fb) else b

    # If we broke out of the first loop (IQUOIN attempt), we fall through to the Secant/Bisection logic above?
    # No, the `break` in the IQUOIN loop just exits that loop.
    # We need to structure the code so the Secant/Bisection logic is executed.
    
    # Let's restructure:
    # 1. Find Bracket.
    # 2. Run Secant/Bisection Hybrid.
    
    # I will rewrite the function to just do Bracketing + Secant/Bisection.
    # It's simpler and robust.

    # Wait, I am editing the existing code.
    # I will replace the entire body of find_root with the new logic.
    
    # --- New Implementation ---
    
    # Reset state for the new approach
    x = float(x0)
    fx = f(x)
    calls = 1
    
    if abs(fx) <= tol:
        return x
        
    # Find Bracket
    a = x
    fa = fx
    b = None
    fb = None
    
    step = 1.0
    # Expand in positive direction
    for i in range(30):
        x_try = a + step
        f_try = f(x_try)
        calls += 1
        if abs(f_try) <= tol:
            return x_try
        if fa * f_try < 0:
            b = x_try
            fb = f_try
            break
        step *= 2.0
        
    if b is None:
        # Expand in negative direction
        step = 1.0
        for i in range(30):
            x_try = a - step
            f_try = f(x_try)
            calls += 1
            if abs(f_try) <= tol:
                return x_try
            if fa * f_try < 0:
                b = x_try
                fb = f_try
                break
            step *= 2.0

    if b is None:
        return a # No bracket found, return best guess

    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa

    # Secant/Bisection Hybrid
    # We use the bracket [a, b].
    # We need a third point for Secant? No, Secant uses 2 points.
    # We can use the current endpoints a, b for Secant.
    # But if f(a) and f(b) are very different, Secant might jump far.
    # The constraint "x_sec inside [a, b]" handles this.
    
    for i in range(100):
        if abs(fa) <= tol:
            return a
        if abs(fb) <= tol:
            return b
            
        # Secant prediction
        if fa != fb:
            x_sec = b - fb * (b - a) / (fb - fa)
        else:
            x_sec = (a + b) / 2.0
            
        # Bisection prediction
        x_bis = (a + b) / 2.0
        
        # Choose step
        if a < x_sec < b:
            x_next = x_sec
        else:
            x_next = x_bis
            
        f_next = f(x_next)
        calls += 1
        
        if abs(f_next) <= tol:
            return x_next
            
        # Update bracket
        if fa * f_next < 0:
            b = x_next
            fb = f_next
        else:
            a = x_next
            fa = f_next
            
        if abs(b - a) < 1e-12:
            return a if abs(fa) < abs(fb) else b

    return a if abs(fa) < abs(fb) else b
