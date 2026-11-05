import time
import numpy as np
from math import lcm
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    n = 30
    # LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Integer weights: w_i = L // i for coefficient c_i multiplying 1/i
    # Note: The task description says coefficient[i-1]/i.
    # So c[0] * 1/1 + c[1] * 1/2 + ... + c[29] * 1/30.
    # Weights correspond to denominators 1 to 30.
    weights = np.array([L // i for i in range(1, n + 1)], dtype=np.int64)
    
    best_abs_sum = np.iinfo(np.int64).max
    best_coeffs = None
    
    # Multi-start local search
    # We use random restarts and hill climbing.
    # The space is small enough that we can afford many iterations.
    
    # Helper to calculate sum
    def calc_sum(coeffs):
        return int(np.dot(coeffs, weights))
    
    # Initialize with a few good heuristics or just random
    # A known good configuration for small N often involves alternating signs or specific patterns.
    # But random restarts are robust.
    
    # To speed up, we can pre-compute weight magnitudes for quick pruning? 
    # Not strictly necessary for N=30, dot product is fast.
    
    max_iterations_per_start = 500
    max_starts = 10000 # We will run as many as time permits
    
    rng = np.random.default_rng(42)
    
    for _ in range(max_starts):
        if time.time() - start_time > deadline:
            break
            
        # Initialize random coefficients
        # Values in {-1, 0, 1}
        # rng.integers(0, 3, size=n) gives 0, 1, 2. Map to -1, 0, 1.
        coeffs = rng.integers(0, 3, size=n).astype(np.int64) - 1
        
        cur_sum = calc_sum(coeffs)
        
        # If initial sum is 0, we need to move away from it immediately or it's invalid.
        # Local search will handle this if we ensure we don't accept 0.
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations_per_start:
            improved = False
            iterations += 1
            
            # Pick a random index to change
            idx = rng.integers(0, n)
            
            # Try all 3 possible values for this index
            # Current value is coeffs[idx]
            cur_val = coeffs[idx]
            
            # Generate permutations of {-1, 0, 1} excluding current value
            # Or just try all 3 and skip if same
            possible_vals = [-1, 0, 1]
            random.shuffle(possible_vals)
            
            for new_val in possible_vals:
                if new_val == cur_val:
                    continue
                
                # Calculate delta
                # new_sum = cur_sum + (new_val - cur_val) * weights[idx]
                delta = (new_val - cur_val) * weights[idx]
                new_sum = cur_sum + delta
                
                if new_sum == 0:
                    continue
                
                abs_new = abs(new_sum)
                abs_cur = abs(cur_sum)
                
                if abs_new < abs_cur:
                    # Accept
                    coeffs[idx] = new_val
                    cur_sum = new_sum
                    improved = True
                    
                    # Update global best
                    if abs_new < best_abs_sum:
                        best_abs_sum = abs_new
                        best_coeffs = coeffs.copy()
                        
                    # If we hit 1, that's the theoretical minimum (numerator 1)
                    # We can't get smaller than 1 for non-zero integer sum.
                    if abs_new == 1:
                        return list(best_coeffs)
                    break # Restart index selection after a successful move? 
                         # Or continue with next index? 
                         # Usually, restarting the scan or picking a new random index is fine.
                         # Breaking here means we picked a new index in the next while iteration.
        
        # If we found 1, return early
        if best_abs_sum == 1:
            break
            
    if best_coeffs is None:
        # Fallback: simple non-zero
        # c = [1, -1, 0, ...] -> 1 - 1/2 = 1/2. Num = L/2.
        # Better: c[0]=1, c[1]=1 -> 1+1/2 = 3/2.
        # Let's just return a random valid one.
        fallback = [0] * 28 + [1, -1]
        # 1/29 - 1/30 = 1/(29*30). Num = L/(29*30).
        # This is small.
        return fallback
        
    return list(best_coeffs)
