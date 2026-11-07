import math
import time
import numpy as np

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 495
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0  # Safety margin for 120s limit
    
    calls = 0
    best_x = float(x0)
    best_f = float('inf')
    f_cache = {}
    
    def safe_f(x):
        nonlocal calls, best_x, best_f
        if calls >= MAX_CALLS:
            return float('inf')
        
        if time.time() > DEADLINE:
            return float('inf')
            
        key = round(float(x), 14)
        
        if key in f_cache:
            val = f_cache[key]
            if abs(val) < best_f:
                best_f = abs(val)
                best_x = float(x)
            return val

        calls += 1
        try:
            val = f(float(x))
            if not math.isfinite(val):
                f_cache[key] = float('inf')
                return float('inf')
            f_cache[key] = val
        except Exception:
            f_cache[key] = float('inf')
            return float('inf')
        
        if abs(val) < best_f:
            best_f = abs(val)
            best_x = float(x)
            
        return val

    x_start = float(x0)
    
    # 1. Check starting point
    fx0 = safe_f(x_start)
    if abs(fx0) <= TOL:
        return x_start

    # 2. Global Coarse-to-Fine Grid Refinement
    # Strategy:
    # - Phase 1: Broad search with low resolution to find a region with small |f(x)|.
    # - Phase 2: Refine the best region with higher resolution.
    # - Phase 3: Further refine.
    # - Phase 4: Final high-resolution check around the best point.
    
    # Scale based on x0 to ensure we don't miss roots far away if x0 is small,
    # but also not waste calls if x0 is large.
    scale = max(1.0, abs(x_start))
    
    # Define grid configurations: (radius_multiplier, num_points)
    # Total points: 100 + 200 + 300 + 400 = 1000? No, too many.
    # We have ~490 calls.
    # Let's use a budget of ~400 calls for the grid.
    
    # Config: (radius, num_points)
    # We want to cover a range of [-R, R] around x_start.
    # Resolution = 2R / N.
    
    # Phase 1: Broad coverage. R=100*scale, N=100. Res=2.
    # Phase 2: Medium. R=10*scale, N=100. Res=0.2.
    # Phase 3: Fine. R=1*scale, N=100. Res=0.02.
    # Phase 4: Very Fine. R=0.1*scale, N=100. Res=0.002.
    
    # But if the root is far, Phase 1 finds a "good enough" x.
    # Then Phase 2 refines around x_start? No, we should refine around the BEST x found.
    
    # Revised Strategy: Adaptive Refinement around Best X
    
    # Initial broad scan
    R = 100.0 * scale
    N = 100
    points = np.linspace(x_start - R, x_start + R, N)
    
    for p in points:
        if time.time() > DEADLINE - 5.0 or calls > MAX_CALLS - 50:
            break
        safe_f(p)
        
    # If we found a very good root, we might stop early?
    # No, we want to minimize calls while ensuring success.
    # If best_f is small, we can stop refining?
    # But oscillatory functions might have a closer root.
    # However, the task is just to find A root.
    
    # Refine around best_x
    # We will perform 3 more refinement steps.
    # Each step reduces the radius by a factor of 10 and increases resolution.
    
    current_radius = 10.0 * scale
    # Wait, the initial scan was R=100. The best_x is somewhere in [-100, 100].
    # So the next radius should be around the best_x.
    
    # Let's do a series of refinements centered on best_x.
    # Radii: 10, 1, 0.1, 0.01
    # Points: 100, 100, 100, 100
    
    radii = [10.0 * scale, 1.0 * scale, 0.1 * scale, 0.01 * scale]
    
    for R_refine in radii:
        if time.time() > DEADLINE - 5.0 or calls > MAX_CALLS - 20:
            break
            
        # If the previous best was already very good, maybe we don't need to scan as wide?
        # But for oscillatory, we might have overshot the root in the coarse grid.
        # The coarse grid step was 200/100 = 2.0 * scale.
        # So the error is at most 1.0 * scale.
        # So refining with R=10*scale is safe.
        
        N_refine = 100
        # Ensure we don't go out of bounds if scale is huge?
        # linspace handles it.
        
        points = np.linspace(best_x - R_refine, best_x + R_refine, N_refine)
        
        for p in points:
            if time.time() > DEADLINE - 2.0 or calls > MAX_CALLS - 5:
                break
            safe_f(p)
            
        # Early exit if we hit tolerance
        if best_f <= TOL:
            return best_x
            
    # Final check: If we still haven't hit TOL, try a very tight local search
    # around best_x with high resolution.
    if best_f > TOL and time.time() < DEADLINE - 2.0 and calls < MAX_CALLS - 10:
        R_final = 0.001 * scale
        N_final = 50
        points = np.linspace(best_x - R_final, best_x + R_final, N_final)
        for p in points:
            if time.time() > DEADLINE - 1.0 or calls > MAX_CALLS:
                break
            safe_f(p)
            
    return best_x
