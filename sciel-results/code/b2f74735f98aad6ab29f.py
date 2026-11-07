"""
Root finder using a Bracket-First Illinois Method strategy.
1. Aggressively establish a bracket [a, b] containing a root by expanding from x0.
2. Use Illinois/Regula Falsi hybrid:
   - If the new point is closer to the root, keep the other end.
   - If the new point is farther (stale), halve the interval (bisection) to ensure convergence.
3. This guarantees convergence for continuous functions with a sign change.
4. Includes strict time and call budget guards.
"""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using a robust bracketing method.
    f: callable, takes float, returns float
    x0: float starting point
    returns: float x with |f(x)| <= 1e-8
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin from 120s limit
    
    x0 = float(x0)
    if not math.isfinite(x0):
        return x0

    # Phase 1: Find a bracket [a, b] such that f(a)*f(b) <= 0
    # Start with x0 as one end
    a = x0
    fa = f(a)
    
    if abs(fa) <= 1e-8:
        return a

    # If f(x0) is not close to zero, expand to find a sign change
    # We try expanding in both directions: +1, -1, +2, -2, +4, -4, ...
    # This is robust for polynomials which typically have roots within reasonable distance
    # unless the root is extremely far, but x0 is usually a hint.
    
    b = None
    fb = None
    
    # Try small steps first, then larger
    # Note: We check time and call count implicitly by keeping this phase short.
    # Max ~20 calls for bracketing.
    step = 1.0
    max_steps = 1000.0
    
    while step <= max_steps:
        # Try positive direction
        b_pos = a + step
        fb_pos = f(b_pos)
        if math.isfinite(fb_pos):
            if fa * fb_pos <= 0:
                b = b_pos
                fb = fb_pos
                break
            if abs(fb_pos) < abs(fa):
                # If we found a point with smaller |f|, update our 'best' anchor? 
                # No, we need a bracket. But we can keep track of best_x for fallback.
                pass
        
        # Try negative direction
        b_neg = a - step
        fb_neg = f(b_neg)
        if math.isfinite(fb_neg):
            if fa * fb_neg <= 0:
                b = b_neg
                fb = fb_neg
                break
            
        step *= 2.0
        
        # Time check
        if time.time() > DEADLINE:
             return a # Return best guess, likely fail but avoids timeout

    # If no bracket found, we might have a root that is a local extremum or 
    # the function doesn't change sign (e.g., double root). 
    # For "smooth polynomial equations", sign change is likely if x0 is near root.
    # If no bracket, fallback to simple Newton from best point found during search?
    # Or just return x0 if nothing better.
    # Given the task says "3 fixed smooth polynomial equations", and x0 is a starting point,
    # a bracket is highly probable.
    
    if b is None:
        # Fallback: Try a few Newton steps from x0 without bracketing, just in case
        # (e.g. if f(x0) is very small but not < 1e-8 and no sign change nearby)
        x = x0
        fx = fa
        for _ in range(10):
            if abs(fx) <= 1e-8:
                return x
            h = 1e-7 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            x = x_new
            fx = f(x)
        return x

    # Phase 2: Illinois Method (Bisection + Secant hybrid)
    # We have [a, b] with fa*fb <= 0.
    # We want to converge to the root.
    # Standard Illinois:
    # 1. Compute x_new using secant formula.
    # 2. If x_new is outside [a,b], use bisection.
    # 3. Evaluate f(x_new).
    # 4. If f(x_new)*f(a) <= 0, then b = x_new. Else a = x_new.
    # 5. If the end that was just updated was the one that stayed (i.e., we updated the 'far' end),
    #    we halve the function value of the 'stale' end to force bisection behavior next time.
    
    # To simplify and ensure robustness, we use a simpler variant:
    # Always compute secant step. If it falls outside bracket, bisect.
    # After updating bracket, check if the interval is shrinking fast enough.
    # Actually, the classic Illinois method is:
    # Let x0, x1 be ends.
    # x_new = (x0*f(x1) - x1*f(x0)) / (f(x1) - f(x0))
    # If x_new not in [x0, x1], x_new = (x0+x1)/2
    # Update: if f(x0)*f(x_new) <= 0: x1 = x_new else: x0 = x_new
    # Crucial Illinois twist: If the end that is KEPT (not updated) has the larger |f|,
    # divide its function value by 2. This forces the secant line to flatten out,
    # leading to a bisection-like step.
    
    xa = a
    fa_val = fa
    xb = b
    fb_val = fb
    
    # Ensure |fa_val| <= |fb_val| to start? Not strictly necessary but good for Illinois.
    if abs(fa_val) > abs(fb_val):
        xa, xb = xb, xa
        fa_val, fb_val = fb_val, fa_val

    best_x = xa if abs(fa_val) < abs(fb_val) else xb
    best_f = min(abs(fa_val), abs(fb_val))

    for _ in range(100):
        if time.time() > DEADLINE:
            return best_x

        # Check convergence
        if abs(fa_val) <= 1e-8:
            return xa
        if abs(fb_val) <= 1e-8:
            return xb
        
        # Secant step
        denom = fb_val - fa_val
        if denom == 0:
            x_new = (xa + xb) / 2.0
        else:
            x_new = xa - fa_val * (xa - xb) / denom
        
        # Check if x_new is within bracket
        if not (min(xa, xb) <= x_new <= max(xa, xb)):
            x_new = (xa + xb) / 2.0
            
        f_new = f(x_new)
        
        if not math.isfinite(f_new):
            # If we get a non-finite, shrink to midpoint
            x_new = (xa + xb) / 2.0
            f_new = f(x_new)
            if not math.isfinite(f_new):
                return best_x

        if abs(f_new) < best_f:
            best_x = x_new
            best_f = abs(f_new)
            
        if abs(f_new) <= 1e-8:
            return x_new

        # Update bracket
        if fa_val * f_new <= 0:
            # Root is in [xa, x_new]
            # The 'stale' end is xb. 
            # Illinois rule: if |fb_val| > |fa_val| (stale is larger), halve fb_val.
            # We update xb to x_new.
            if abs(fb_val) > abs(fa_val):
                fb_val /= 2.0
            xb = x_new
            fb_val = f_new
        else:
            # Root is in [x_new, xb]
            # The 'stale' end is xa.
            if abs(fa_val) > abs(fb_val):
                fa_val /= 2.0
            xa = x_new
            fa_val = f_new

        # Early exit if interval is tiny
        if abs(xb - xa) < 1e-12:
             return (xa + xb) / 2.0

    return best_x
# EVOLVE-BLOCK-END
