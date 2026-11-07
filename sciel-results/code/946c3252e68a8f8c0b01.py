import math
import time
import numpy as np
from scipy.optimize import brentq, minimize_scalar

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin under 120s
    TARGET = 1e-8
    MAX_CALLS = 480     # Safety margin under 500

    call_count = 0
    # Cache to avoid duplicate calls
    cache = {}

    def f_counted(x):
        nonlocal call_count
        if call_count >= MAX_CALLS:
            raise RuntimeError("Max calls exceeded")
        
        x_key = round(x, 15) # Avoid floating point noise in key
        if x_key in cache:
            return cache[x_key]
            
        call_count += 1
        val = f(x)
        if not math.isfinite(val):
            cache[x_key] = float('nan')
            return float('nan')
        cache[x_key] = val
        return val

    x0 = float(x0)
    fx0 = f_counted(x0)
    if abs(fx0) <= TARGET:
        return x0

    # List of (x, f(x)) sorted by x
    # We will expand from x0
    points = [(x0, fx0)]
    
    # Step sizes to try
    # We want to cover a reasonable range efficiently.
    # Oscillatory functions often have periods in [0.1, 10].
    # Let's try steps: 0.01, 0.1, 1, 10
    # For each step, we probe outwards.
    
    step_sizes = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    
    bracket = None
    
    for h in step_sizes:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Probe right
        prev_x = x0
        prev_fx = fx0
        for i in range(1, 20):
            x_new = x0 + i * h
            # Check if we already have this point (from previous step size or direction)
            # The cache handles the call count, but we need to update 'points' list logic
            # Actually, for bracketing, we just need to know if sign changed between consecutive probes.
            # If we jump scales, the "consecutive" logic is tricky.
            # Better: Maintain a sorted list of all evaluated points and check neighbors.
            
            f_new = f_counted(x_new)
            
            if abs(f_new) <= TARGET:
                return x_new
                
            # Add to points
            points.append((x_new, f_new))
            
            # Check for sign change with the closest evaluated point to the left
            # Since we are adding points in increasing order for this direction, 
            # the previous point in this specific loop is the immediate neighbor 
            # IF we haven't skipped over it with a larger step.
            # But we are iterating h. 
            # Example: h=0.1, i=1 -> x0+0.1. h=0.01, i=10 -> x0+0.1.
            # The cache prevents duplicate calls.
            # To find a bracket, we need two points a, b such that f(a)*f(b) < 0.
            # We can check the new point against ALL existing points? That's O(N^2).
            # N is small (< 100), so O(N^2) is fine.
            
            found_bracket = False
            # Sort points by x to find neighbors?
            # Or just check against the point we came from?
            # If we use geometric expansion, we might skip a root.
            # E.g. f(0)=1, f(1)=-1, f(2)=1. Root in (0,1) and (1,2).
            # If we step by 1.5: x0=0, x1=1.5 (f=0.5?), x2=3.
            # We might miss the sign change if the function oscillates fast.
            
            # Robust approach: 
            # 1. Collect a set of candidate points.
            # 2. Sort them.
            # 3. Check adjacent pairs for sign changes.
            
            # Let's just check the new point against the "previous" point in the current sequence.
            # But "previous" depends on the step size.
            
            # Simpler: Just check the new point against x0 and the last point evaluated in this direction.
            # If f(x0) and f(x_new) have different signs, there is a root in between.
            # This is guaranteed by IVT if continuous.
            # If they have the same sign, there might still be an even number of roots.
            # But for oscillatory, we usually want the first sign change.
            
            if f_new * prev_fx < 0:
                bracket = (prev_x, x_new)
                found_bracket = True
                break
            
            # Also check if x_new and x0 have different signs? 
            # If f(x0)=1, f(x1)=-1, f(x2)=1. 
            # x1 and x0 change sign.
            # If f(x0)=1, f(x1)=1, f(x2)=-1. 
            # x2 and x1 change sign.
            # So checking against the immediate previous point in the scan is sufficient 
            # IF the scan is dense enough not to skip a full period.
            
            prev_x = x_new
            prev_fx = f_new
            
        if bracket:
            break
            
        # Probe left
        prev_x = x0
        prev_fx = fx0
        for i in range(1, 20):
            x_new = x0 - i * h
            f_new = f_counted(x_new)
            
            if abs(f_new) <= TARGET:
                return x_new
                
            points.append((x_new, f_new))
            
            if f_new * prev_fx < 0:
                bracket = (x_new, prev_x)
                break
            
            prev_x = x_new
            prev_fx = f_new
            
        if bracket:
            break

    if bracket:
        a, b = bracket
        # Ensure a < b
        if a > b:
            a, b = b, a
        try:
            root = brentq(f_counted, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if abs(f_counted(root)) <= TARGET:
                return root
        except Exception:
            pass

    # Fallback: Minimize f(x)^2
    # Use a bounded search around x0
    # If we have many points, we can find the min |f| point and search around it.
    best_x = x0
    best_f = fx0
    for x, fx in points:
        if abs(fx) < abs(best_f):
            best_f = fx
            best_x = x
            
    if abs(best_f) <= TARGET:
        return best_x
        
    # Try local minimization around best_x
    # Width depends on the step size that found the best point? 
    # Just use a reasonable default.
    widths = [1e-2, 1e-1, 1, 10]
    for w in widths:
        if time.time() - start_time > TIME_LIMIT:
            break
        try:
            res = minimize_scalar(lambda x: f_counted(x)**2, bounds=(best_x - w, best_x + w), method='bounded', options={'xatol': 1e-12})
            if res.success and res.fun <= TARGET**2:
                return res.x
        except Exception:
            pass
            
    return best_x
