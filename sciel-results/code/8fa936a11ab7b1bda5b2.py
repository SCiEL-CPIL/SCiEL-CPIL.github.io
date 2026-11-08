import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder using a multi-stage strategy:
    1. Coarse-to-fine grid search to locate a local minimum of |f(x)|.
       This avoids the fragility of sign-change bracketing on oscillatory functions
       where the function might not change sign (e.g., touching zero) or where
       bracketing requires many calls.
    2. Newton-Raphson refinement from the best grid point.
       Newton's method converges quadratically, drastically reducing the number of
       calls needed to reach 1e-8 precision compared to bisection or Illinois.
    3. Fallback to Bisection if Newton fails to converge or diverges.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin for 120s limit
    
    x0 = float(x0)
    
    # Cache to avoid redundant calls if the same x is evaluated multiple times
    # (unlikely with floats, but good practice)
    cache = {}
    
    def f_cached(x):
        if x in cache:
            return cache[x]
        val = f(x)
        cache[x] = val
        return val
    
    # Initial check
    fx0 = f_cached(x0)
    if abs(fx0) <= 1e-8:
        return x0
        
    best_x = x0
    best_f = abs(fx0)
    
    # Phase 1: Coarse-to-Fine Grid Search
    # We search in expanding rings around x0.
    # For each radius, we sample points.
    # If we find a point with |f| < 1e-8, we return immediately.
    # If we find a point with |f| significantly smaller than current best, we update best.
    
    # Radii to check: 1e-2, 1e-1, 1, 10, 100, 1000
    # Number of points per ring: 16, 32, 64, 128, 256, 512
    # Total calls approx: 1 + 16 + 32 + 64 + 128 + 256 + 512 = 1009? No, we stop early.
    # We should be smarter. We don't need to check all rings if we find a very small value.
    
    search_configs = [
        (1e-2, 8),
        (1e-1, 16),
        (1.0, 32),
        (10.0, 64),
        (100.0, 128),
        (1000.0, 256)
    ]
    
    # We will keep track of the best bracket found during the scan as well, 
    # in case Newton fails.
    a_bracket = None
    b_bracket = None
    
    for radius, n_points in search_configs:
        if time.time() > deadline - 5.0:
            break
            
        # Generate angles
        angles = [2 * math.pi * i / n_points for i in range(n_points)]
        
        prev_x = None
        prev_f = None
        
        for angle in angles:
            x_try = x0 + radius * math.cos(angle)
            # Also check sine component? No, 1D. Just x0 + radius * cos(angle) covers -radius to +radius.
            # Actually, cos(angle) for 0..2pi covers [-1, 1].
            # So x_try is in [x0 - radius, x0 + radius].
            
            # Skip if too close to x0 or previous point to avoid redundant work
            if abs(x_try - x0) < 1e-15:
                continue
            if prev_x is not None and abs(x_try - prev_x) < 1e-15:
                continue
                
            f_try = f_cached(x_try)
            
            if abs(f_try) < best_f:
                best_f = abs(f_try)
                best_x = x_try
                
            if abs(f_try) <= 1e-8:
                return x_try
                
            # Check for sign change with previous point to maintain a bracket
            if prev_f is not None and prev_f * f_try < 0:
                # Found a bracket
                a_bracket = prev_x
                b_bracket = x_try
                # We can break out of this ring if we have a good bracket, 
                # but let's finish the ring to see if there's a better (narrower) bracket?
                # Or just break to save calls?
                # Breaking is safer for call count.
                break
                
            prev_x = x_try
            prev_f = f_try
            
            # If we found a bracket, we can stop searching this ring
            if a_bracket is not None:
                break
                
        # If we found a bracket in this ring, we might want to refine it immediately?
        # Or just proceed to Newton from best_x?
        # Newton from best_x is likely better if best_x is close to root.
        # If best_x is not close, Newton might diverge.
        # Let's proceed to Phase 2.
        
        # Optimization: If best_f is already very small (e.g., < 1e-4), 
        # we can skip larger radii.
        if best_f < 1e-4:
            break

    # Phase 2: Newton-Raphson Refinement
    # Start from best_x found in Phase 1.
    # If best_x is x0 and we didn't find a better point, we still try Newton.
    
    x = best_x
    fx = f_cached(x)
    
    # If we have a bracket, we can use it to ensure Newton stays in bounds or fallback.
    # But standard Newton is fast.
    
    for i in range(50):
        if time.time() > deadline - 2.0:
            break
            
        if abs(fx) <= 1e-8:
            return x
            
        # Compute derivative using central difference
        h = 1e-6
        # Adjust h if x is very large or small
        if abs(x) > 1e6:
            h = 1e-2
        elif abs(x) < 1e-6:
            h = 1e-10
            
        f_plus = f_cached(x + h)
        f_minus = f_cached(x - h)
        dfdx = (f_plus - f_minus) / (2 * h)
        
        if abs(dfdx) < 1e-12:
            # Derivative is too small, Newton will fail. 
            # Break and try Bisection if we have a bracket.
            break
            
        x_new = x - fx / dfdx
        f_new = f_cached(x_new)
        
        # Check for divergence
        if not math.isfinite(x_new) or not math.isfinite(f_new):
            break
            
        # If we have a bracket, ensure x_new stays within it?
        # Newton can jump out of the bracket. 
        # If it jumps out, we can bisect instead.
        if a_bracket is not None:
            a_min, a_max = min(a_bracket, b_bracket), max(a_bracket, b_bracket)
            if x_new < a_min or x_new > a_max:
                # Newton jumped out. Use Bisection step.
                x_mid = (a_bracket + b_bracket) / 2
                f_mid = f_cached(x_mid)
                
                # Update bracket
                if f_mid * f_cached(a_bracket) < 0:
                    b_bracket = x_mid
                else:
                    a_bracket = x_mid
                    
                x = x_mid
                fx = f_mid
                continue
                
        x = x_new
        fx = f_new
        
    # Phase 3: Fallback Bisection if Newton didn't converge
    if abs(fx) > 1e-8:
        if a_bracket is None:
            # No bracket found. Return best_x found so far.
            # This might fail, but it's the best we have.
            return best_x
            
        a = a_bracket
        b = b_bracket
        fa = f_cached(a)
        fb = f_cached(b)
        
        # Ensure opposite signs
        if fa * fb > 0:
            # This shouldn't happen if we found it via sign change, but check just in case
            # If signs are same, we can't bisect. Return best_x.
            return best_x
            
        for i in range(100):
            if time.time() > deadline - 0.5:
                break
                
            x_mid = (a + b) / 2
            f_mid = f_cached(x_mid)
            
            if abs(f_mid) <= 1e-8:
                return x_mid
                
            if fa * f_mid < 0:
                b = x_mid
                fb = f_mid
            else:
                a = x_mid
                fa = f_mid
                
            if abs(b - a) < 1e-15:
                break
                
        return (a + b) / 2

    return x
# EVOLVE-BLOCK-END
