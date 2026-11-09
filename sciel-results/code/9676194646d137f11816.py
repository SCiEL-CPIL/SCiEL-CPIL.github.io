import math
import time

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Uses Illinois bisection for robustness and speed on oscillatory functions.
    """
    start_time = time.time()
    deadline = start_time + 5.0  # 5 seconds is plenty for <500 calls
    
    # Tolerance for root
    tol = 1e-12  # Tighter internal tolerance to ensure final check passes
    max_iter = 200
    
    x0 = float(x0)
    f0 = f(x0)
    if abs(f0) <= tol:
        return x0

    # Try to find a bracket [a, b] such that f(a) * f(b) < 0
    # Start with a small step and expand
    # Adaptive step size based on x0
    base_step = 1e-4 * max(1.0, abs(x0))
    
    a = x0
    fa = f0
    b = None
    fb = None
    
    # Search for a sign change
    # Try a few steps in both directions
    steps = [1, -1, 2, -2, 4, -4, 8, -8, 16, -16]
    for s in steps:
        x_try = x0 + s * base_step
        f_try = f(x_try)
        
        if not math.isfinite(f_try):
            continue
            
        if abs(f_try) <= tol:
            return x_try
            
        if fa * f_try < 0:
            # Found a bracket
            if abs(fa) < abs(f_try):
                a = x0
                b = x_try
                fa = f0
                fb = f_try
            else:
                a = x_try
                b = x0
                fa = f_try
                fb = f0
            break
    
    if b is not None:
        # Use Illinois Bisection Method
        # This is a derivative-free method with guaranteed convergence
        # It modifies bisection by shrinking the longer subinterval more aggressively
        # when the function value at the new point has the same sign as one endpoint
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # If f(a) is small enough, return a
            if abs(fa) <= tol:
                return a
            # If f(b) is small enough, return b
            if abs(fb) <= tol:
                return b
            
            # Check if we need to bisect
            # If the interval is small, return midpoint
            if abs(b - a) < tol:
                return (a + b) / 2.0
            
            # Illinois method: 
            # Compute midpoint
            mid = (a + b) / 2.0
            f_mid = f(mid)
            
            if not math.isfinite(f_mid):
                # If non-finite, try a different point
                mid = a + (b - a) * 0.25
                f_mid = f(mid)
                if not math.isfinite(f_mid):
                    mid = a + (b - a) * 0.75
                    f_mid = f(mid)
                    if not math.isfinite(f_mid):
                        break
            
            if abs(f_mid) <= tol:
                return mid
            
            # Update bracket
            if fa * f_mid < 0:
                # Root is in [a, mid]
                b = mid
                fb = f_mid
                # Illinois: shrink the other side more
                # We'll handle this in the next iteration by adjusting how we choose mid
            else:
                # Root is in [mid, b]
                a = mid
                fa = f_mid
            
            # Illinois adjustment: 
            # If the new midpoint had the same sign as the previous midpoint,
            # shrink the longer subinterval
            # For simplicity, we use a standard Illinois variant:
            # If f_mid has same sign as f_prev_mid, shrink the interval on that side by half
            
            # A simpler Illinois variant:
            # Always bisect, but if the function is flat, adjust
            # Let's use the standard Illinois algorithm properly
            
            # Actually, let's use a cleaner Illinois implementation:
            # We maintain a bracket and always bisect, but we modify the 
            # interval shrinking based on which side is "worse"
            
            # For now, let's just use bisection with early termination
            # This is robust and guaranteed to converge

        # Return the point with smallest |f(x)|
        # We need to track the best point
        # Let's redo this with proper tracking

        # Restart with proper Illinois bisection
        a = x0
        fa = f0
        b = None
        fb = None
        
        # Re-search for bracket (we might have lost it)
        steps = [1, -1, 2, -2, 4, -4, 8, -8, 16, -16]
        for s in steps:
            x_try = x0 + s * base_step
            f_try = f(x_try)
            
            if not math.isfinite(f_try):
                continue
                
            if abs(f_try) <= tol:
                return x_try
                
            if fa * f_try < 0:
                if abs(fa) < abs(f_try):
                    a = x0
                    b = x_try
                    fa = f0
                    fb = f_try
                else:
                    a = x_try
                    b = x0
                    fa = f_try
                    fb = f0
                break
        
        if b is not None:
            # Proper Illinois Bisection
            # Reference: Illinois method for root finding
            # At each step:
            # 1. Compute midpoint m = (a+b)/2
            # 2. Evaluate f(m)
            # 3. If f(a)*f(m) < 0, set b = m
            #    Else set a = m
            # 4. If the new interval is on the same side as the previous,
            #    shrink the other endpoint by half
            
            prev_a = a
            prev_b = b
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                    
                if abs(fa) <= tol:
                    return a
                if abs(fb) <= tol:
                    return b
                
                if abs(b - a) < tol:
                    return (a + b) / 2.0
                
                mid = (a + b) / 2.0
                f_mid = f(mid)
                
                if not math.isfinite(f_mid):
                    # Try different points
                    candidates = [a + (b-a)*0.25, a + (b-a)*0.75, a + (b-a)*0.1, a + (b-a)*0.9]
                    found = False
                    for cand in candidates:
                        f_cand = f(cand)
                        if math.isfinite(f_cand):
                            mid = cand
                            f_mid = f_cand
                            found = True
                            break
                    if not found:
                        break
                
                if abs(f_mid) <= tol:
                    return mid
                
                if fa * f_mid < 0:
                    # Root in [a, mid]
                    old_b = b
                    b = mid
                    fb = f_mid
                    # Illinois: if we're on the left side, shrink the right
                    if b - a < old_b - a:  # We moved b left
                        # Shrink a towards b
                        a = a + (b - a) * 0.5
                        fa = f(a)
                        if abs(fa) <= tol:
                            return a
                else:
                    # Root in [mid, b]
                    old_a = a
                    a = mid
                    fa = f_mid
                    # Illinois: if we're on the right side, shrink the left
                    if b - a < b - old_a:  # We moved a right
                        # Shrink b towards a
                        b = b - (b - a) * 0.5
                        fb = f(b)
                        if abs(fb) <= tol:
                            return b
            
            # Return the better of a and b
            if abs(fa) < abs(fb):
                return a
            else:
                return b

    else:
        # No bracket found. Use Secant Method with safeguards.
        # Initialize second point
        x_prev = x0
        f_prev = f0
        
        # Try a few steps to find a better starting point for Secant
        # We want a point where the slope is not zero and we are moving towards a root
        x_curr = x0 + base_step
        f_curr = f(x_curr)
        
        if abs(f_curr) <= tol:
            return x_curr
            
        best_x = x_prev if abs(f_prev) < abs(f_curr) else x_curr
        best_f = min(abs(f_prev), abs(f_curr))
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            denom = f_curr - f_prev
            if abs(denom) < 1e-15:
                # Slope is zero, break
                break
                
            x_next = x_curr - f_curr * (x_curr - x_prev) / denom
            f_next = f(x_next)
            
            if not math.isfinite(f_next) or not math.isfinite(x_next):
                break
                
            if abs(f_next) < best_f:
                best_x = x_next
                best_f = abs(f_next)
                
            if abs(f_next) <= tol:
                return x_next
                
            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_next
            f_curr = f_next
            
        return best_x
