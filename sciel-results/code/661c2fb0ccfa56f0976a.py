"""
Root finder using Trust-Region Inverse Quadratic Interpolation (IQI).
Strategy:
1. Establish a bracket [a, b] containing a root by expanding from x0.
2. Maintain three points (a, b, c) to define an inverse quadratic interpolant.
3. Compute the next step using the inverse quadratic formula.
4. Enforce trust-region constraints:
   - If the IQI step is outside the bracket [min(a,b), max(a,b)], use Bisection.
   - If the IQI step is less than half the bisection step (too conservative), use Bisection.
   - This ensures global convergence while retaining the fast local convergence of IQI.
5. Update the points, dropping the least useful point (usually the one with the largest |f| 
   among the older endpoints, or simply maintaining a sliding window of the last 3 distinct 
   points that form a valid bracket structure).
"""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    x0 = float(x0)
    if not math.isfinite(x0):
        return 0.0

    # Track best solution found
    best_x = x0
    best_f = f(x0)
    if abs(best_f) <= 1e-8:
        return best_x

    # 1. Find a bracket [a, b] such that f(a)*f(b) <= 0
    a = x0
    fa = best_f
    
    b = None
    fb = None
    
    # Search for bracket
    # Try small steps first, then expand
    steps = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
    found_bracket = False
    
    for step in steps:
        if time.time() > deadline:
            return best_x
        for sign in [1, -1]:
            b = a + sign * step
            if not math.isfinite(b):
                continue
            fb = f(b)
            if not math.isfinite(fb):
                continue
            
            if abs(fb) <= 1e-8:
                return b
            
            if fa * fb <= 0:
                found_bracket = True
                break
        
        if found_bracket:
            break
    
    if not found_bracket:
        # Fallback: if no bracket found, try a denser search in a wider range
        # This is unlikely for smooth polynomials with reasonable x0, but good for safety.
        for scale in [0.01, 0.1, 1.0, 10.0, 100.0]:
             for sign in [1, -1]:
                 b = x0 + sign * scale
                 fb = f(b)
                 if not math.isfinite(fb):
                     continue
                 if abs(fb) <= 1e-8: return b
                 if fa * fb <= 0:
                     found_bracket = True
                     break
             if found_bracket: break
                 
        if not found_bracket:
            return best_x

    # 2. Iteration: Trust-Region IQI
    # State:
    # a, fa: current left endpoint (or one of the bracket endpoints)
    # b, fb: current right endpoint (or the other bracket endpoint)
    # c, fc: the previous point used for quadratic interpolation (initially same as a or b)
    
    # Initialize c to a. The first step will effectively be secant/bisection.
    # We need 3 distinct points to do quadratic. 
    # Strategy: 
    # 1. Do a bisection step to get a 3rd point.
    # 2. Then use IQI.
    
    c = a
    fc = fa
    
    # Ensure a and b are ordered for clarity in bracket logic, though IQI formula is symmetric.
    # Let's keep a and b as the current bracket endpoints.
    
    max_iter = 100
    
    for i in range(max_iter):
        if time.time() > deadline:
            return best_x
            
        # Check current endpoints
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
            
        # Update best
        if abs(fa) < abs(best_f):
            best_f = fa
            best_x = a
        if abs(fb) < abs(best_f):
            best_f = fb
            best_x = b
        if abs(fc) < abs(best_f):
            best_f = fc
            best_x = c

        # Determine the bracket interval [lo, hi]
        # We need to identify which pair of points forms the bracket.
        # fa and fb have opposite signs.
        # c is a previous point.
        
        # The "active" bracket is [a, b].
        # We want to find the next point d.
        
        # Bisection candidate
        m = (a + b) / 2.0
        fm = f(m)
        
        if not math.isfinite(fm):
            # If bisection point is bad, we are stuck or function is weird.
            # Try shrinking bracket? No, if bisection fails, we can't proceed safely.
            # Return best.
            break
            
        if abs(fm) <= 1e-8:
            return m
            
        # Now we have 3 points: (a, fa), (b, fb), (m, fm).
        # We want to use IQI to predict the next step.
        # The inverse quadratic interpolant passes through (fa, a), (fb, b), (fm, m).
        # We solve for x such that p(x) = 0, where p is the interpolant of y vs x?
        # No, Inverse Quadratic Interpolation (IQI) fits a parabola through (fa, a), (fb, b), (fm, m) 
        # in the form x = q(f). We want the root of q(f)=0? No, we want the x where f(x)=0.
        # So we fit a quadratic polynomial q(y) that passes through (fa, a), (fb, b), (fm, m).
        # Then the predicted next x is q(0).
        
        # Formula for x_new (denoted as d) using Lagrange interpolation for x(y) at y=0:
        # d = (fa*fa*(b-m) / ((fa-fb)*(fa-fm)) + fb*fb*(m-a) / ((fb-fa)*(fb-fm)) + fm*fm*(a-b) / ((fm-fa)*(fm-fb)))
        
        # Let's compute the terms carefully to avoid division by zero.
        # If any denominators are close to zero, fall back to bisection.
        
        d = None
        try:
            denom1 = (fa - fb) * (fa - fm)
            denom2 = (fb - fa) * (fb - fm)
            denom3 = (fm - fa) * (fm - fb)
            
            # Check for small denominators
            if abs(denom1) > 1e-15 and abs(denom2) > 1e-15 and abs(denom3) > 1e-15:
                term1 = (fa * fa * (b - m)) / denom1
                term2 = (fb * fb * (m - a)) / denom2
                term3 = (fm * fm * (a - b)) / denom3
                d = term1 + term2 + term3
        except ZeroDivisionError:
            d = None
            
        # Trust Region Constraints:
        # 1. d must be within [min(a,b), max(a,b)]
        # 2. d must be at least half the bisection step away from the center? 
        #    Standard Brent's method rule: if IQI step is less than half the bisection step, use bisection.
        #    This prevents slow convergence when the quadratic is flat.
        
        min_ab = min(a, b)
        max_ab = max(a, b)
        
        use_bisection = False
        
        if d is None:
            use_bisection = True
        elif not (min_ab < d < max_ab):
            use_bisection = True
        else:
            # Check if d is too close to the center (conservative)
            # Distance from center to d vs Distance from center to endpoint
            dist_d_to_center = abs(d - m)
            dist_center_to_endpoint = abs(m - a) # or abs(m-b), they are same
            if dist_d_to_center < 0.5 * dist_center_to_endpoint:
                use_bisection = True
        
        if use_bisection:
            next_x = m
            next_f = fm
        else:
            next_x = d
            next_f = f(d)
            if not math.isfinite(next_f):
                # If IQI step was bad, fall back to the bisection point we already evaluated
                next_x = m
                next_f = fm

        # Update Bracket
        # We have points a, b, m (or d).
        # We know fa*fb <= 0.
        # We need to replace either a or b with the new point (next_x) such that the new bracket still contains the root.
        
        if fa * next_f <= 0:
            # Root is in [a, next_x]
            # Replace b with next_x? 
            # Wait, if fa*next_f <= 0, then a and next_x bracket the root.
            # So the new bracket is [a, next_x].
            # The old b is discarded.
            # But we need to keep 3 points for the next IQI step.
            # The points are (a, fa), (next_x, next_f), and... what is the 3rd point?
            # It should be the one that is "closest" to the root or most recent?
            # In Brent's method, we keep the point that was just replaced? No.
            # We keep the 3 points: the two bracket endpoints and the one that was just "tested" but rejected?
            # Actually, standard Brent's keeps:
            # a, b (bracket)
            # c (previous best guess)
            
            # Let's follow the standard update:
            # If we used bisection (next_x == m):
            #   New bracket is either [a, m] or [m, b].
            #   The 3rd point c should be the one that is NOT in the new bracket? 
            #   Or the one that is "older"?
            
            # Let's simplify: 
            # We maintain (a, fa), (b, fb) as the bracket.
            # And (c, fc) as the "memory" point.
            
            # If we pick new bracket [a, next_x]:
            #   b becomes next_x, fb becomes next_f.
            #   What happens to c?
            #   If c was b (old), then c is now "outside". 
            #   If c was a or m, we need to decide.
            
            # Robust strategy:
            # The 3 points for the NEXT iteration should be:
            # 1. New a
            # 2. New b
            # 3. The point among {old a, old b, new point} that is NOT new a or new b, 
            #    OR if all 3 are new, pick the one with smallest |f|?
            #    Actually, for IQI, we just need 3 distinct points.
            #    Let's keep the point that was just "rejected" from the bracket if it exists, 
            #    otherwise keep the previous c.
            
            # Simpler approach: 
            # Always keep the last 3 points evaluated that are distinct.
            # But we must ensure the bracket is valid.
            
            # Let's use the standard "Illinois" or "Brent" update logic for the memory point.
            # If we shrink [a,b] to [a, next_x]:
            #   The point 'b' is no longer in the bracket.
            #   So the new set of points for interpolation is {a, next_x, b(old)}.
            #   So: 
            #   new_a = a
            #   new_b = next_x
            #   new_c = b (old)
            
            # If we shrink [a,b] to [next_x, b]:
            #   The point 'a' is no longer in the bracket.
            #   So the new set is {next_x, b, a(old)}.
            #   new_a = next_x
            #   new_b = b
            #   new_c = a (old)
            
            # Let's implement this.
            
            old_a, old_fa = a, fa
            old_b, old_fb = b, fb
            
            if fa * next_f <= 0:
                # Bracket is [a, next_x]
                # a stays a. b becomes next_x. c becomes old b.
                b = next_x
                fb = next_f
                c = old_b
                fc = old_fb
            else:
                # Bracket is [next_x, b]
                # b stays b. a becomes next_x. c becomes old a.
                a = next_x
                fa = next_f
                c = old_a
                fc = old_fa
                
        else:
            # This case is handled by the if/else above. 
            # Wait, the condition `if fa * next_f <= 0` covers both cases?
            # If fa * next_f > 0, then fb * next_f must be <= 0 (since fa*fb<=0).
            # So yes, one of the two brackets is valid.
            pass

    return best_x

# EVOLVE-BLOCK-END
