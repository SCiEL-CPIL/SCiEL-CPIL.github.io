"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from collections import defaultdict

MAX_VAL = 10_000_000
TIME_LIMIT = 82  # seconds, safety margin before 90s kill

def _sigma_sieve(n):
    """Compute sigma(k) for all k in [1, n] using a sieve-like approach."""
    # sigma(n) = sum of divisors. Use additive sieve: for each d, add d to sigma[d], sigma[2d], ...
    sigma = np.zeros(n + 1, dtype=np.int64)
    for d in range(1, n + 1):
        sigma[d::d] += d
    return sigma

def construct():
    start_time = time.time()
    
    # Compute sigma for all n up to MAX_VAL using vectorized sieve
    sigma_arr = _sigma_sieve(MAX_VAL)
    
    # Build fibers: map sigma_value -> list of n with that sigma
    # To save memory/time, only process values where sigma is "small" enough
    # Actually, let's just build the full mapping
    # Group indices by sigma value
    fiber_map = defaultdict(list)
    for n in range(1, MAX_VAL + 1):
        s = int(sigma_arr[n])
        fiber_map[s].append(n)
    
    # For each fiber, find pair (a, b) minimizing |2a - 3b| / (2b)
    # where a != b (to avoid trivial a=b which gives error 0.5, but also a/b=1 not 3/2)
    # Actually a=b gives |1 - 3/2| = 1/2, which is worse than 1/3
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # For a given fiber (sorted list of values), use two-pointer:
    #   For each b in fiber, find a in fiber closest to 3b/2
    #   Since fiber is sorted, we can binary search for 3b/2 in the fiber
    
    best_num = None
    best_den = None
    best_pair = (11, 6)  # fallback: sigma(11)=12, sigma(6)=12
    
    # Process fibers, prioritizing larger fibers (more chance of good pairs)
    # But also need to be time-efficient
    # Sort fibers by size descending to find good candidates early
    fibers_sorted = sorted(fiber_map.items(), key=lambda x: len(x[1]), reverse=True)
    
    for s_val, vals in fibers_sorted:
        if time.time() - start_time > TIME_LIMIT - 3:
            break
        
        # vals is already sorted (we added in order)
        # For each b in vals, find a in vals closest to 3*b/2
        # Use two-pointer or binary search
        n_vals = len(vals)
        if n_vals < 2:
            continue
        
        # Two-pointer approach: 
        # We want to minimize |2a - 3b|/(2b) for a, b in vals, a != b
        # For fixed b, optimal a is the element in vals closest to 3b/2
        # Since vals is sorted, for each b we can binary search
        
        # But doing this for all b in large fibers could be slow
        # Let's use a smarter two-pointer:
        # Iterate b from smallest to largest, maintain pointer a_idx such that vals[a_idx] ≈ 3*vals[b]/2
        
        a_idx = 0
        for b_idx in range(n_vals):
            b = vals[b_idx]
            target = 3 * b / 2.0
            
            # Advance a_idx while vals[a_idx] < target
            while a_idx < n_vals - 1 and vals[a_idx] < target:
                a_idx += 1
            
            # Check vals[a_idx] and vals[a_idx-1] (if valid)
            candidates_a = set()
            if a_idx < n_vals:
                candidates_a.add(vals[a_idx])
            if a_idx > 0:
                candidates_a.add(vals[a_idx - 1])
            
            for a in candidates_a:
                if a == b:
                    continue
                if 2 * a == 3 * b:
                    continue  # error = 0, invalid
                error_num = abs(2 * a - 3 * b)
                error_den = 2 * b
                
                if best_num is None:
                    best_num = error_num
                    best_den = error_den
                    best_pair = (a, b)
                else:
                    # Compare error_num/error_den < best_num/best_den
                    if error_num * best_den < best_num * error_den:
                        best_num = error_num
                        best_den = error_den
                        best_pair = (a, b)
        
        # Early termination if we found a very good pair
        if best_num is not None and best_num <= 1:
            # Error is at most 1/(2b), which is very small
            break
    
    a, b = best_pair
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
