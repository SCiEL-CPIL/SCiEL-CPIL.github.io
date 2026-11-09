import math
import time
import scipy.optimize as opt

def find_root(f, x0):
    START_TIME = time.time()
    TIME_LIMIT = 100.0
    MAX_CALLS = 490
    calls = 0
    cache = {}
    best_x = x0
    best_f = float('inf')
    found = False

    def safe_f(x):
        nonlocal calls, best_x, best_f, found
        if found:
            return 0.0 # Prevent further calls if we already have a solution
        if calls >= MAX_CALLS:
            return float('nan')
        if time.time() - START_TIME > TIME_LIMIT:
            return float('nan')
        
        # Cache check
        # Using x directly as key is safe for floats if they are identical bits
        if x in cache:
            val = cache[x]
        else:
            calls += 1
            try:
                val = f(x)
                val = float(val)
            except Exception:
                val = float('nan')
            cache[x] = val
            
        if math.isfinite(val):
            abs_val = abs(val)
            if abs_val < best_f:
                best_f = abs_val
                best_x = x
                if best_f <= 1e-8:
                    found = True
        return val

    # Initial evaluation
    fx0 = safe_f(x0)
    if found:
        return best_x

    # Helper to check bracket
    def check_bracket(xa, xb):
        fa = cache.get(xa, safe_f(xa))
        fb = cache.get(xb, safe_f(xb))
        if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0:
            return (xa, xb)
        return None

    # Phase 1: Fine local scan
    # Scan from x0-1.0 to x0+1.0 with step 0.05
    # This is 40 steps in each direction.
    h = 0.05
    max_range = 1.0
    
    # We need to track the previous point to check brackets between consecutive points
    prev_x = x0
    prev_fx = fx0
    
    # Positive direction
    for i in range(1, int(max_range/h) + 1):
        x_new = x0 + i * h
        fx_new = safe_f(x_new)
        if found:
            return best_x
            
        if math.isfinite(prev_fx) and math.isfinite(fx_new) and prev_fx * fx_new < 0:
            bracket = (prev_x, x_new)
            # Refine
            return refine_root(bracket, safe_f, cache, best_x, best_f)
            
        prev_x = x_new
        prev_fx = fx_new
        
    # Negative direction
    prev_x = x0
    prev_fx = fx0
    for i in range(1, int(max_range/h) + 1):
        x_new = x0 - i * h
        fx_new = safe_f(x_new)
        if found:
            return best_x
            
        if math.isfinite(prev_fx) and math.isfinite(fx_new) and prev_fx * fx_new < 0:
            bracket = (x_new, prev_x)
            return refine_root(bracket, safe_f, cache, best_x, best_f)
            
        prev_x = x_new
        prev_fx = fx_new

    # Phase 2: Coarser wider scan
    # If we didn't find it in +/- 1.0, try +/- 10.0 with step 0.1
    # Then +/- 100.0 with step 1.0
    # We reuse the last evaluated point as the start of the new segment
    
    # Current "frontier" points
    pos_frontier = x0 + max_range
    neg_frontier = x0 - max_range
    pos_fx = cache.get(pos_frontier, safe_f(pos_frontier))
    neg_fx = cache.get(neg_frontier, safe_f(neg_frontier))
    
    if found:
        return best_x

    configs = [
        (0.1, 10.0),   # Step 0.1, Range 10.0
        (0.5, 50.0),   # Step 0.5, Range 50.0
        (2.0, 200.0)   # Step 2.0, Range 200.0
    ]
    
    for step, target_range in configs:
        # Positive
        x_curr = pos_frontier
        fx_curr = pos_fx
        steps_needed = int((target_range - (x_curr - x0)) / step) + 1
        for i in range(steps_needed):
            if found:
                return best_x
            x_new = x_curr + step
            if x_new - x0 > target_range:
                break
            fx_new = safe_f(x_new)
            if math.isfinite(fx_curr) and math.isfinite(fx_new) and fx_curr * fx_new < 0:
                bracket = (x_curr, x_new)
                return refine_root(bracket, safe_f, cache, best_x, best_f)
            x_curr = x_new
            fx_curr = fx_new
            
        pos_frontier = x_curr
        pos_fx = fx_curr
        
        # Negative
        x_curr = neg_frontier
        fx_curr = neg_fx
        steps_needed = int((target_range - (x0 - x_curr)) / step) + 1
        for i in range(steps_needed):
            if found:
                return best_x
            x_new = x_curr - step
            if x0 - x_new > target_range:
                break
            fx_new = safe_f(x_new)
            if math.isfinite(fx_curr) and math.isfinite(fx_new) and fx_curr * fx_new < 0:
                bracket = (x_new, x_curr)
                return refine_root(bracket, safe_f, cache, best_x, best_f)
            x_curr = x_new
            fx_curr = fx_new
            
        neg_frontier = x_curr
        neg_fx = fx_curr

    # If no bracket found, return best_x
    return best_x

def refine_root(bracket, safe_f, cache, best_x, best_f):
    a, b = bracket
    try:
        # Use brentq
        root = opt.brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
        if math.isfinite(root):
            fr = safe_f(root)
            if math.isfinite(fr) and abs(fr) <= 1e-8:
                return root
            # If brentq result is not good enough, try bisection
    except Exception:
        pass
        
    # Fallback: Bisection
    fa = cache.get(a, safe_f(a))
    fb = cache.get(b, safe_f(b))
    
    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    for _ in range(60):
        mid = (a + b) / 2.0
        fm = safe_f(mid)
        if not math.isfinite(fm):
            # If mid is bad, try shrinking towards the finite side?
            # Or just break.
            break
            
        if abs(fm) < 1e-8:
            return mid
            
        if fa * fm <= 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
            
    # Return the midpoint of the final interval or the best point
    # The best point might have been updated by safe_f
    # We should return the one with smallest |f|
    # safe_f updates a global best_x? No, it's local to find_root.
    # We need to pass best_x/best_f back or make them nonlocal in refine_root.
    # For simplicity, let's just return the midpoint or the endpoint with smaller |f|.
    # Actually, safe_f in find_root updates best_x.
    # But refine_root doesn't have access to update the outer best_x easily unless we pass it.
    # Let's just return the candidate with min |f| from a, b, mid.
    
    candidates = [a, b]
    if 'mid' in locals():
        candidates.append(mid)
        
    best_local = min(candidates, key=lambda x: abs(cache.get(x, safe_f(x))))
    return best_local
