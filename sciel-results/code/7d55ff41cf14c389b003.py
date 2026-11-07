# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Robust scalar root finder using a combination of:
    1. Initial sampling to bracket the root
    2. Bisection for guaranteed convergence
    3. Secant method for faster convergence once close
    4. Fallback to bisection if secant diverges
    
    Strategy:
    - First, try to find a bracket [a, b] where f(a) and f(b) have opposite signs
    - Use bisection to narrow down, then switch to secant for speed
    - Track call count and stop early if |f(x)| <= 1e-8
    """
    x = float(x0)
    tol = 1e-8
    max_calls = 450  # Stay well under 500
    
    # We'll track the best x seen so far (smallest |f(x)|)
    best_x = x
    best_fx = f(x)
    call_count = 1
    
    # If already good enough
    if abs(best_fx) <= tol:
        return best_x
    
    # Try to find a bracket around x0
    # Sample points at increasing distances
    a = x
    fa = best_fx
    
    # Try expanding search for a sign change
    found_bracket = False
    step = 1.0
    for i in range(20): # Increased from 10 to 20 for wider search
        if call_count >= max_calls - 10:
            break
        b = a + step
        fb = f(b)
        call_count += 1
        if abs(fb) < abs(best_fx):
            best_x = b
            best_fx = fb
        if abs(fb) <= tol:
            return b
        if fa * fb < 0:
            found_bracket = True
            break
        # Also try the other side
        b2 = a - step
        fb2 = f(b2)
        call_count += 1
        if abs(fb2) < abs(best_fx):
            best_x = b2
            best_fx = fb2
        if abs(fb2) <= tol:
            return b2
        if fa * fb2 < 0:
            a = b2
            fa = fb2
            b = x
            fb = best_fx
            found_bracket = True
            break
        step *= 2.0
    
    if found_bracket:
        # We have a bracket [a, b] with fa*fb < 0
        # Use bisection + secant hybrid
        x1, x2 = a, b
        f1, f2 = fa, fb
        # Ensure f1 < 0 and f2 > 0 for consistent tracking
        if f1 > 0:
            x1, x2 = x2, x1
            f1, f2 = f2, f1
        
        # Bisection to get close, then secant
        for i in range(30):
            if call_count >= max_calls:
                break
            
            # Calculate Secant point
            if f2 != f1:
                xsec = x2 - f2 * (x2 - x1) / (f2 - f1)
            else:
                xsec = (x1 + x2) / 2.0
            
            # Check if Secant point is valid
            # 1. Must be finite
            # 2. Must not be too close to endpoints (to avoid stagnation)
            # 3. Prefer if it's inside the bracket, but allow outside if it's not too far?
            # Standard Brent-like: Only use Secant if it's inside [x1, x2].
            # If outside, use Bisection.
            
            use_secant = False
            if math.isfinite(xsec):
                # Check if inside bracket
                if min(x1, x2) <= xsec <= max(x1, x2):
                    # Check distance from endpoints
                    dist_to_x1 = abs(xsec - x1)
                    dist_to_x2 = abs(xsec - x2)
                    bracket_width = abs(x2 - x1)
                    # If it's very close to an endpoint, use bisection to ensure progress
                    if dist_to_x1 > 1e-4 * bracket_width and dist_to_x2 > 1e-4 * bracket_width:
                        use_secant = True
                        x_next = xsec
                    else:
                        x_next = (x1 + x2) / 2.0
                else:
                    x_next = (x1 + x2) / 2.0
            else:
                x_next = (x1 + x2) / 2.0
            
            if call_count >= max_calls:
                break
            
            f_next = f(x_next)
            call_count += 1
            
            if abs(f_next) < abs(best_fx):
                best_x = x_next
                best_fx = f_next
            if abs(f_next) <= tol:
                return x_next
            
            # Update bracket
            if f_next * f1 < 0:
                x2 = x_next
                f2 = f_next
            else:
                x1 = x_next
                f1 = f_next
                
            # Convergence check on interval
            if abs(x2 - x1) < 1e-12:
                break
        
        # Final check
        if abs(best_fx) <= tol:
            return best_x
        return best_x
    
    else:
        # No bracket found. Use a local search approach.
        # Try secant method with adaptive step, falling back to gradient descent-like
        x_curr = x
        f_curr = best_fx
        
        # Try a few secant iterations with two nearby points
        h = 1e-4 * max(1.0, abs(x_curr))
        x_prev = x_curr - h
        if call_count < max_calls:
            f_prev = f(x_prev)
            call_count += 1
            if abs(f_prev) < abs(best_fx):
                best_x = x_prev
                best_fx = f_prev
            if abs(f_prev) <= tol:
                return x_prev
        
        for i in range(40):
            if call_count >= max_calls:
                break
            if f_curr == f_prev:
                break
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            
            # If x_new is not finite or too far, fall back
            if not math.isfinite(x_new) or abs(x_new - x_curr) > 10 * max(1.0, abs(x_curr)):
                x_new = x_curr - 0.5 * h
                h *= 0.5
                if abs(h) < 1e-15:
                    break
            
            if call_count >= max_calls:
                break
            f_new = f(x_new)
            call_count += 1
            if abs(f_new) < abs(best_fx):
                best_x = x_new
                best_fx = f_new
            if abs(f_new) <= tol:
                return x_new
            
            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_new
            f_curr = f_new
        
        return best_x

# EVOLVE-BLOCK-END
