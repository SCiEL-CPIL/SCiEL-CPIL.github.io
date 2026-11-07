"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    deadline = time.time() + 145.0
    tol = 1e-8
    max_iter = 50

    x = float(x0)
    
    # Initial evaluation
    try:
        fx = f(x)
    except:
        return x
    if not math.isfinite(fx):
        return x
    if abs(fx) <= tol:
        return x
    
    # 1. Bracketing Phase
    # We need two points a, b such that f(a)*f(b) <= 0.
    # Start with a small step, expand exponentially if no sign change.
    a = x
    fa = fx
    h = 1e-4 * max(1.0, abs(x))
    b = a + h
    
    bracket_found = False
    fb = None
    for i in range(50):
        if time.time() > deadline:
            break
        try:
            fb = f(b)
        except:
            break
        if not math.isfinite(fb):
            break
        if abs(fb) <= tol:
            return b
        
        if fa * fb <= 0:
            bracket_found = True
            break
        
        # If no sign change, expand the bracket.
        # We move b further away from a.
        # To keep a and b distinct and cover more ground, we can shift the window.
        # Simple strategy: double the step size and move b.
        # If f(a) and f(b) have same sign, the root is not between them.
        # We keep 'a' fixed? No, better to keep the point with smaller |f| or just expand.
        # Let's just expand the range.
        # If f(a)*f(b) > 0, we don't know which side the root is on.
        # However, for smooth polynomials, expanding usually finds a sign change.
        # We will try expanding to the right first.
        b = a + h * (2 ** (i+1))
        
    if not bracket_found:
        # Try expanding to the left if right failed?
        # Or just fall back to a more robust search if bracketing fails.
        # Given the "3 fixed smooth polynomial" constraint, bracketing should work.
        # If it fails, we might be stuck in a region where f doesn't change sign (unlikely for root finding tasks unless no root exists or multiple roots with even multiplicity).
        # Assuming standard root finding, let's try a quick secant without bracket if this fails, or just return best guess.
        # But let's try expanding in the other direction.
        b = a - h
        for i in range(50):
            if time.time() > deadline:
                break
            try:
                fb = f(b)
            except:
                break
            if not math.isfinite(fb):
                break
            if abs(fb) <= tol:
                return b
            
            if fa * fb <= 0:
                bracket_found = True
                break
            
            b = a - h * (2 ** (i+1))
            
    if not bracket_found:
        # Fallback: If we couldn't bracket, use a simple secant/Newton hybrid without guarantee.
        # This is a safety net.
        x_curr = x
        f_curr = fx
        for _ in range(30):
            if time.time() > deadline:
                break
            # Estimate derivative
            h_step = 1e-7 * max(1.0, abs(x_curr))
            try:
                f_next = f(x_curr + h_step)
            except:
                break
            if not math.isfinite(f_next):
                break
            if abs(f_next) <= tol:
                return x_curr + h_step
            
            d = (f_next - f_curr) / h_step
            if abs(d) < 1e-14:
                break
            
            x_new = x_curr - f_curr / d
            if not math.isfinite(x_new):
                break
            
            # Damping
            step = x_new - x_curr
            if abs(step) > 1.0:
                x_new = x_curr + (1.0 if step > 0 else -1.0)
            
            x_curr = x_new
            try:
                f_curr = f(x_curr)
            except:
                break
            if not math.isfinite(f_curr):
                break
            if abs(f_curr) <= tol:
                return x_curr
        
        return x_curr

    # 2. Safeguarded Secant Method
    # We have a bracket [a, b] with fa, fb.
    # We maintain two points x0, x1 and their function values.
    # Initially x0=a, x1=b.
    x0 = a
    x1 = b
    f0 = fa
    f1 = fb
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Check convergence
        if abs(f0) <= tol:
            return x0
        if abs(f1) <= tol:
            return x1
            
        # Calculate Secant step
        if f1 != f0:
            x_sec = x0 - f0 * (x1 - x0) / (f1 - f0)
        else:
            # Degenerate case, use bisection
            x_sec = 0.5 * (x0 + x1)
            
        # Safeguard: Check if x_sec is within the bracket [min(x0,x1), max(x0,x1)]
        # Note: The bracket shrinks as we update points.
        # Actually, standard safeguarded secant keeps the bracket [a, b] separate from the secant points?
        # No, usually we update the bracket.
        # Let's stick to a simpler model:
        # We have a bracket [a, b].
        # We calculate x_new.
        # If x_new is inside [a, b], we evaluate f(x_new).
        #   If f(a)*f(x_new) <= 0, new bracket is [a, x_new].
        #   Else new bracket is [x_new, b].
        #   Then we use secant with the two points of the NEW bracket?
        #   Or do we keep the previous point for secant?
        #   To maximize speed, we should use the secant prediction based on the two most recent points,
        #   but ensure it stays in the bracket. If it goes out, we bisect.
        
        # Let's implement the "Safeguarded Secant" where we maintain a bracket [a, b] and two points x_prev, x_curr.
        # Actually, the simplest robust version is:
        # 1. Start with x0, x1 from bracket.
        # 2. Compute x2 from secant.
        # 3. If x2 is outside [min(x0,x1), max(x0,x1)], use bisection point.
        # 4. Evaluate f(x2).
        # 5. Update bracket: keep the half that contains the sign change.
        # 6. Set x0 to the point that was NOT the new midpoint? No.
        #    Secant needs two points. The best two points are usually the two most recent,
        #    but if we bisect, we lose the secant history.
        
        # Let's use a specific algorithm:
        # Maintain a bracket [a, b].
        # Maintain two points p, q (initially a, b).
        # Step:
        #   1. Calculate x_new from secant(p, q).
        #   2. If x_new is not in [a, b], set x_new = (a+b)/2.
        #   3. Evaluate f_new.
        #   4. Update bracket [a, b] to the sub-interval containing the root.
        #   5. Update p, q:
        #      - If x_new was secant, we can keep q and set p=x_new? Or keep the two points with smallest |f|?
        #      - To be safe and fast, let's just use the two points of the NEW bracket for the next secant step.
        #      - This effectively makes it a "Secant with Bisection Fallback" where the secant is always calculated from the current bracket endpoints.
        #      - This is slower than true secant (which reuses history) but much more robust and simpler.
        #      - However, true secant reuses history. Let's try to reuse history.
        
        # Let's try:
        # x_prev, f_prev, x_curr, f_curr.
        # Bracket [a, b].
        
        # To keep it simple and efficient:
        # We will just use the bracket endpoints for the secant calculation.
        # This is effectively a "Bisection with Secant Prediction".
        
        x_low = min(a, b)
        x_high = max(a, b)
        f_low = f(a) if a == x_low else f(b) # Wait, we have fa, fb stored?
        # We have fa, fb from the bracketing phase, but they change as we update a, b.
        # Let's maintain a, b, fa, fb.
        
        # Re-initialize loop variables to be clear
        # We are in the loop, so a, b, fa, fb are current bracket.
        
        # Calculate Secant Point
        if fb != fa:
            x_candidate = a - fa * (b - a) / (fb - fa)
        else:
            x_candidate = 0.5 * (a + b)
            
        # Check if candidate is inside bracket
        if (x_candidate <= x_low or x_candidate >= x_high):
            x_candidate = 0.5 * (a + b)
            
        # Evaluate
        try:
            f_candidate = f(x_candidate)
        except:
            break
        if not math.isfinite(f_candidate):
            break
            
        if abs(f_candidate) <= tol:
            return x_candidate
            
        # Update Bracket
        if fa * f_candidate <= 0:
            b = x_candidate
            fb = f_candidate
        else:
            a = x_candidate
            fa = f_candidate
            
        # Optional: If the interval is very small, break
        if abs(b - a) < 1e-12:
            break
            
    # Return the best point
    # We don't have f(a) or f(b) stored in a way that compares easily if we just returned a or b.
    # But we can check both one last time? No, limit calls.
    # Just return the midpoint of the final bracket or the last evaluated point.
    # Since we update a, b, fa, fb, the last evaluated point is either a or b.
    # We can return the one with smaller |f| if we had it, but we only have fa, fb.
    # Let's return the midpoint to be safe? No, return the one with smaller |f|.
    if abs(fa) < abs(fb):
        return a
    else:
        return b

# EVOLVE-BLOCK-END
