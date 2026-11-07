"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    for i in range(10):
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
            # Check if midpoint is good
            mid = (x1 + x2) / 2.0
            if call_count >= max_calls - 1:
                break
            fmid = f(mid)
            call_count += 1
            if abs(fmid) < abs(best_fx):
                best_x = mid
                best_fx = fmid
            if abs(fmid) <= tol:
                return mid
            if fmid * f1 < 0:
                x2 = mid
                f2 = fmid
            else:
                x1 = mid
                f1 = fmid
            
            # Try secant step if we have two points
            if call_count >= max_calls - 5:
                break
            if f2 != f1:
                # Secant estimate
                xsec = x2 - f2 * (x2 - x1) / (f2 - f1)
                # Only accept if it's within the bracket
                if min(x1, x2) < xsec < max(x1, x2):
                    if call_count >= max_calls:
                        break
                    fsec = f(xsec)
                    call_count += 1
                    if abs(fsec) < abs(best_fx):
                        best_x = xsec
                        best_fx = fsec
                    if abs(fsec) <= tol:
                        return xsec
                    if fsec * f1 < 0:
                        x2 = xsec
                        f2 = fsec
                    else:
                        x1 = xsec
                        f1 = fsec
                # If secant went outside, just continue bisection
        
        # Final check
        if abs(best_fx) <= tol:
            return best_x
        # Return the best we have
        # Do one more bisection step if possible
        if call_count < max_calls:
            mid = (x1 + x2) / 2.0
            fmid = f(mid)
            call_count += 1
            if abs(fmid) < abs(best_fx):
                best_x = mid
                best_fx = fmid
            if abs(fmid) <= tol:
                return mid
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
