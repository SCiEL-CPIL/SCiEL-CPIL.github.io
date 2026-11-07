import math
import time
import numpy as np

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 495  # Safety margin
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0  # 120s limit, 5s margin

    calls = 0
    best_x = float(x0)
    best_f = float('inf')
    f_cache = {}
    
    # Golden ratio constant
    GR = (1 + math.sqrt(5)) / 2
    INV_GR = 1.0 / GR

    def safe_f(x):
        nonlocal calls, best_x, best_f
        if calls >= MAX_CALLS:
            return float('inf')
        if time.time() > DEADLINE:
            return float('inf')
            
        key = float(x)
        # Use a small epsilon for cache key to handle floating point noise
        # but keep it distinct enough for search progress.
        # Actually, exact match is fine if we pass exact floats from the algorithm.
        if key in f_cache:
            val = f_cache[key]
            if val < best_f:
                best_f = val
                best_x = key
            return val

        calls += 1
        try:
            val = f(key)
            if not math.isfinite(val):
                f_cache[key] = float('inf')
                return float('inf')
            f_cache[key] = val
        except Exception:
            f_cache[key] = float('inf')
            return float('inf')
        
        if val < best_f:
            best_f = val
            best_x = key
            
        return val

    def objective(x):
        return safe_f(x) ** 2

    x_start = float(x0)
    
    # Check starting point
    fx0 = safe_f(x_start)
    if fx0 ** 2 <= TOL ** 2:
        return x_start

    # Strategy: Global Adaptive Search
    # We want to find the global minimum of f(x)^2.
    # We start with a wide interval and narrow it down.
    # If the minimum is not at the center, we expand the interval.
    
    # Initial scale estimation
    # If x0 is 0, scale is 1. If x0 is large, scale is relative.
    # However, for oscillatory functions, the "period" matters.
    # A safe initial range is [-10, 10] scaled by x0 magnitude if x0 is large,
    # but often roots are near 0 or within O(1).
    
    # Let's use a hierarchical expansion approach.
    # 1. Try a local window around x0.
    # 2. If no good root, expand.
    
    # Define a function to perform Golden Section Search on [a, b]
    def golden_section_search(a, b, max_iter=60):
        nonlocal best_x, best_f, calls
        if calls >= MAX_CALLS - 5:
            return best_x
            
        # Ensure a < b
        if a > b:
            a, b = b, a
            
        # Initial points
        # c = a + (b - a) / GR
        # d = b - (b - a) / GR
        c = a + (b - a) * INV_GR
        d = b - (b - a) * INV_GR
        
        fc = objective(c)
        fd = objective(d)
        
        # If we hit tolerance immediately, return
        if fc <= TOL**2: return c
        if fd <= TOL**2: return d
        
        for _ in range(max_iter):
            if time.time() > DEADLINE - 2.0:
                break
            if calls >= MAX_CALLS - 5:
                break
            
            if fc < fd:
                # Minimum is in [a, d]
                b = d
                d = c
                fd = fc
                c = a + (b - a) * INV_GR
                fc = objective(c)
            else:
                # Minimum is in [c, b]
                a = c
                c = d
                fc = fd
                d = b - (b - a) * INV_GR
                fd = objective(d)
                
            # Check convergence
            if abs(b - a) < 1e-12:
                break
            if fc <= TOL**2: return c
            if fd <= TOL**2: return d

        # Return the best of c, d
        if fc < fd:
            return c
        else:
            return d

    # Hierarchy of scales
    # We try scales that cover increasingly larger domains.
    # Scale 1: Local to x0
    # Scale 2: Moderate
    # Scale 3: Large
    # Scale 4: Very Large
    
    # Determine base scale
    base_scale = 1.0
    if abs(x_start) > 1.0:
        base_scale = abs(x_start)
    elif abs(x_start) < 1e-6:
        base_scale = 1.0 # Don't go too small, might miss nearby roots in oscillatory funcs
    
    # We will test intervals centered at x0
    # [x0 - S, x0 + S]
    
    scales = [1.0, 10.0, 100.0, 1000.0]
    
    # Also consider that the root might be far from x0.
    # The problem says "x0 is a float starting point".
    # Usually this implies the root is somewhat near, but for oscillatory, it could be anywhere.
    # However, scanning the whole real line is impossible.
    # The "Global" aspect here is robustness to local minima within the window.
    
    # To be more robust, we can also try shifting the window if x0 is not 0.
    # But let's stick to centered windows first.
    
    # Optimization: If x0 is 0, the first window is [-1, 1].
    # If the function is oscillatory with period ~1, this might contain multiple roots.
    # Golden section on f^2 finds the GLOBAL min in that window.
    
    for scale in scales:
        if time.time() > DEADLINE - 5.0:
            break
        if calls >= MAX_CALLS - 10:
            break
            
        half_width = base_scale * scale
        a = x_start - half_width
        b = x_start + half_width
        
        # Perform search
        candidate = golden_section_search(a, b)
        
        # Check if we found a root
        if candidate is not None:
            # Verify the root
            if abs(safe_f(candidate)) <= TOL:
                return candidate
        
        # If we didn't find a root in this window, we expand.
        # But wait, golden_section_search returns the best x in the window.
        # If that best x doesn't satisfy the tolerance, we need to expand.
        
        # Early exit if we are using too many calls
        if calls > MAX_CALLS * 0.5:
            # If we've used half the budget and haven't found a root,
            # we should be more aggressive or stop.
            # For now, we continue to the next scale if we have budget.
            pass

    # If we get here, we didn't find a root in the tested windows.
    # This is a failure case for the current strategy.
    # We return the best x found so far.
    return best_x
