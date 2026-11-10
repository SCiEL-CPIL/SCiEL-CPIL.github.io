"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find the longest interval [start, start+length-1] within [1, 1008]
    such that for every x, y in the interval (including x=y), x*y % 1009 != 1.
    
    Approach:
    1. Precompute modular inverses.
    2. Identify constraints:
       - Self-inverses x (x^2=1) impose that if L <= x, R < x.
       - Pairs (a, b) with a < b and inv(a)=b impose that if L <= a, R < b.
    3. For each L, calculate the maximum valid R by taking the minimum limit
       among all constraints with threshold >= L.
    4. Maximize R - L + 1.
    """
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
        
    # Identify constraints
    # Constraint: (threshold, limit) means if L <= threshold, then R <= limit.
    constraints = []
    
    # Self-inverses
    for x in range(1, n + 1):
        if inv[x] == x:
            # If L <= x, R must be < x
            constraints.append((x, x - 1))
            
    # Distinct pairs
    # To avoid duplicates, only consider x < inv[x]
    for x in range(1, n + 1):
        y = inv[x]
        if x < y:
            # If L <= x, the interval contains x.
            # To be valid, it must not contain y.
            # So R must be < y.
            constraints.append((x, y - 1))
            
    # Bucket constraints by threshold
    # We iterate L from n down to 1.
    # At step L, we activate all constraints with threshold == L.
    buckets = {i: [] for i in range(1, n + 1)}
    for thr, lim in constraints:
        if thr in buckets:
            buckets[thr].append(lim)
            
    best_start = 1
    best_len = 0
    
    # global_min_limit tracks the minimum limit among all active constraints
    # Active constraints are those with threshold >= current L.
    global_min_limit = n  # Initial limit is n (1008)
    
    for L in range(n, 0, -1):
        # Activate constraints with threshold L
        for lim in buckets[L]:
            if lim < global_min_limit:
                global_min_limit = lim
                
        R_L = global_min_limit
        # R_L is the maximum R such that [L, R_L] is valid
        # But R_L must be at least L for a non-empty interval?
        # If R_L < L, the interval is empty/invalid.
        
        if R_L >= L:
            length = R_L - L + 1
            if length > best_len:
                best_len = length
                best_start = L
                
    return [best_start, best_len]
# EVOLVE-BLOCK-END
