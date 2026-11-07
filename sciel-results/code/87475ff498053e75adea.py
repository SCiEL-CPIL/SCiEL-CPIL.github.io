import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 118.0  # Safety margin for 120s limit
    
    calls = 0
    best_x = x0
    best_f_val = float('inf')
    cache = {}
    
    def _f(x):
        nonlocal calls, best_x, best_f_val
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError("Time limit exceeded")
        
        # Check cache
        if x in cache:
            return cache[x]
            
        try:
            val = float(f(x))
        except Exception:
            cache[x] = float('nan')
            return float('nan')
            
        if not math.isfinite(val):
            cache[x] = float('nan')
            return float('nan')
            
        calls += 1
        cache[x] = val
        
        abs_val = abs(val)
        if abs_val < best_f_val:
            best_f_val = abs_val
            best_x = x
            
        return val

    # Initial evaluation
    x0 = float(x0)
    fx0 = _f(x0)
    
    if abs(fx0) <= 1e-8:
        return x0

    # Strategy:
    # 1. Perform a dense sweep over a logarithmically expanding domain.
    #    This ensures that for oscillatory functions, we don't jump over roots
    #    by using a fixed number of points per order of magnitude of distance.
    # 2. Collect all brackets [x_i, x_{i+1}] where f(x_i) and f(x_{i+1}) have opposite signs.
    # 3. Refine the best bracket (smallest width or smallest |f| at endpoints) using Illinois method.
    
    # We will generate a list of x points to evaluate.
    # We start with x0.
    # Then we expand in both directions.
    
    # To be efficient, we'll generate points in a specific order:
    # Closest to x0 first.
    
    # Let's define a set of "rings" or scales.
    # Scale k has radius R_k = base * (growth ** k)
    # We sample N points in [-R_k, R_k] not already sampled.
    # Actually, simpler:
    # Generate a sorted list of candidate points.
    # We want to cover a range [x0 - R, x0 + R].
    
    # Let's try a specific grid structure.
    # Points: x0
    # Then x0 +/- d, x0 +/- 2d, ...
    # But for oscillatory functions, we need relative density.
    
    # Alternative:
    # Evaluate at x0 + delta * i where delta is small initially.
    # If no bracket found, increase range.
    
    # Let's use a "concentric" sampling.
    # We'll maintain a list of evaluated points sorted by x.
    # We'll insert new points to maintain a maximum gap size relative to the current range.
    
    # Simpler approach that is robust:
    # 1. Check x0.
    # 2. Check x0 +/- 1e-3, +/- 2e-3, ... up to some limit. (Linear expansion)
    #    This might be too slow for large roots.
    # 3. Check x0 +/- 1, +/- 2, ...
    # 4. Check x0 +/- 10, +/- 20, ...
    
    # Let's combine:
    # Phase 1: Local search with step 1e-3, 1e-2, 1e-1, 1.
    # For each step size s, check x0 +/- s, x0 +/- 2s, ... up to 10 steps?
    # This is O(N) calls.
    
    # Let's try a "Multi-Resolution" sweep.
    # We define a set of points to check.
    # We prioritize points closer to x0.
    
    points_to_check = []
    
    # Add x0
    # We'll generate points in shells.
    # Shell 0: x0
    # Shell 1: x0 +/- 1e-3
    # Shell 2: x0 +/- 2e-3, x0 +/- 1e-2
    # ...
    
    # Actually, a very effective strategy for "hidden oscillatory" is to assume the period is unknown but likely not extremely small or large.
    # A dense grid over a reasonable range is safest.
    
    # Let's try:
    # Range: [-100, 100] around x0?
    # Points: 500 points? No, we have budget for ~450 calls.
    # If we use 400 calls for sampling, we can afford a grid of 400 points.
    # But where to place them?
    
    # If the root is far, uniform grid over [-100, 100] with 400 points has step 0.5.
    # If oscillation period is 0.1, we miss it.
    # If we use log-spaced points, we catch large scale features but miss small scale far away.
    
    # Hybrid:
    # 1. Fine grid near x0: [-1, 1] with 50 points (step 0.04).
    # 2. Medium grid: [-10, 10] with 50 points (step 0.4).
    # 3. Coarse grid: [-100, 100] with 50 points (step 4).
    # 4. Very coarse: [-1000, 1000] with 20 points.
    
    # Total calls: ~170 for sampling.
    # Remaining ~280 for refinement.
    
    # Let's implement this.
    
    def add_point(x):
        if x not in cache:
            return True
        return False

    # We will generate a list of unique x values to evaluate.
    candidates = []
    
    # Helper to add range
    def add_range(start, end, num_points):
        if num_points <= 1:
            candidates.append(start)
            return
        step = (end - start) / (num_points - 1)
        for i in range(num_points):
            candidates.append(start + i * step)
            
    # Define the grids
    # Grid 1: Local
    add_range(x0 - 1.0, x0 + 1.0, 50)
    # Grid 2: Medium
    add_range(x0 - 10.0, x0 + 10.0, 50)
    # Grid 3: Wide
    add_range(x0 - 100.0, x0 + 100.0, 50)
    # Grid 4: Very Wide
    add_range(x0 - 1000.0, x0 + 1000.0, 20)
    # Grid 5: Huge (just in case)
    add_range(x0 - 10000.0, x0 + 10000.0, 10)
    
    # Sort candidates by distance from x0 to prioritize local search
    candidates = sorted(set(candidates), key=lambda x: abs(x - x0))
    
    # Evaluate candidates until we find a bracket or run out of budget/time
    # We need to keep track of evaluated points to find brackets.
    # A bracket is formed by two adjacent evaluated points (in the sorted order of ALL evaluated points)
    # that have opposite signs.
    
    evaluated = {x0: fx0}
    
    # We'll iterate through candidates.
    # To find brackets efficiently, we can just store (x, f(x)) and sort at the end?
    # Or insert into a sorted structure?
    # Since N is small (~180), sorting at the end is fine.
    
    # But we want to stop early if we find a good bracket?
    # "Good" bracket: small width.
    # If we find a bracket with width < 1e-3, we can stop sampling and refine.
    
    brackets = []
    
    # To detect brackets as we go, we'd need a dynamic sorted list.
    # Given the small N, let's just evaluate all candidates (or until budget) and then find brackets.
    # This is simpler and robust.
    
    # Budget for sampling: 200 calls.
    # We have ~180 candidates. So we evaluate all of them if budget allows.
    
    for x in candidates:
        if calls >= 250: # Leave 200 for refinement
            break
        if time.time() - start_time > TIME_LIMIT:
            break
            
        fx = _f(x)
        evaluated[x] = fx
        
        # If we hit a root directly, return
        if abs(fx) <= 1e-8:
            return x
            
    # Now find brackets from evaluated points
    sorted_x = sorted(evaluated.keys())
    
    # Check adjacent pairs
    for i in range(len(sorted_x) - 1):
        x1 = sorted_x[i]
        x2 = sorted_x[i+1]
        f1 = evaluated[x1]
        f2 = evaluated[x2]
        
        if math.isnan(f1) or math.isnan(f2):
            continue
            
        if f1 * f2 < 0:
            # Found a bracket
            brackets.append((x1, x2, f1, f2))
            
    # If no brackets found, return best_x
    if not brackets:
        return best_x
        
    # Select the best bracket
    # Heuristic: smallest width (x2 - x1)
    # Or smallest |f1| + |f2|?
    # Smallest width is usually better for convergence speed.
    brackets.sort(key=lambda b: b[1] - b[0])
    
    # Refine the best bracket
    a, b, fa, fb = brackets[0]
    
    # Illinois Method
    x1, x2 = a, b
    f1, f2 = fa, fb
    
    # Ensure f1 < 0 < f2 or f1 > 0 > f2?
    # Illinois works regardless of sign, as long as they are opposite.
    
    for _ in range(50):
        if calls >= 450:
            break
        if time.time() - start_time > TIME_LIMIT:
            break
            
        if abs(f1) <= 1e-8:
            return x1
        if abs(f2) <= 1e-8:
            return x2
            
        # Illinois step
        # Replace the point with larger |f| with the midpoint
        if abs(f1) > abs(f2):
            x1 = 0.5 * (x1 + x2)
            f1 = _f(x1)
        else:
            x2 = 0.5 * (x1 + x2)
            f2 = _f(x2)
            
        if math.isnan(f1) or math.isnan(f2):
            break
            
        if f1 * f2 > 0:
            # Bracket lost, break
            break
            
    # Return the better of x1, x2
    if abs(f1) < abs(f2):
        return x1
    else:
        return x2

# EVOLVE-BLOCK-END
