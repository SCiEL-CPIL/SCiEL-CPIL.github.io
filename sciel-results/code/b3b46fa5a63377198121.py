"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    
    # Precompute all arithmetic progressions within [1, 256]
    # An AP is defined by (start, step, length) where all terms are in [1, n]
    # We need to minimize the max absolute sum over all such APs.
    # For efficiency, we can precompute the list of indices for each AP.
    
    # Generate all APs
    aps = []
    for start in range(1, n + 1):
        for step in range(1, n + 1):
            # Generate terms: start, start+step, start+2*step, ... <= n
            terms = list(range(start, n + 1, step))
            if terms:
                aps.append(terms)
    
    # Convert to 0-based indices for array indexing
    # But the problem says [1, 256], so index i corresponds to position i-1 in 0-based
    # Let's store 0-based indices
    ap_indices = []
    for terms in aps:
        ap_indices.append([t - 1 for t in terms])
    
    num_aps = len(ap_indices)
    
    # For fast evaluation, we can precompute a matrix or use vectorized operations
    # But num_aps might be large. Let's estimate.
    # For step=1: n + (n-1) + ... + 1 = n(n+1)/2 = 32896
    # For step=2: ~n/2 + ~n/4 + ... ~ n
    # Total is roughly O(n^2 log n) or similar.
    # Let's just build a list of numpy arrays or a sparse representation.
    
    # Alternative: Use a dense matrix where M[i, j] = 1 if j-th AP contains i-th element.
    # Then sum = M @ signs.
    # But M is sparse.
    
    # Let's use a different approach: 
    # 1. Start with a good initial guess (e.g., random or heuristic)
    # 2. Use local search: flip a sign if it reduces the max absolute sum.
    
    # Initial sign assignment
    rng = random.Random(42)
    signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
    
    # Function to compute max absolute sum
    def max_abs_sum(signs):
        max_sum = 0
        for idx in ap_indices:
            s = np.sum(signs[idx])
            abs_s = abs(s)
            if abs_s > max_sum:
                max_sum = abs_s
        return max_sum
    
    current_max = max_abs_sum(signs)
    
    # Local search
    # For each position, try flipping it and see if it reduces the max.
    # To speed up, we can keep track of the current sums for each AP.
    
    # Precompute for each position, which APs contain it.
    pos_to_aps = [[] for _ in range(n)]
    for ap_id, idx in enumerate(ap_indices):
        for i in idx:
            pos_to_aps[i].append(ap_id)
    
    # Current sums for each AP
    ap_sums = np.zeros(num_aps, dtype=np.int32)
    for ap_id, idx in enumerate(ap_indices):
        ap_sums[ap_id] = np.sum(signs[idx])
    
    current_max = np.max(np.abs(ap_sums))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try flipping each position
        for i in range(n):
            if time.time() > deadline:
                break
            # Flip sign at i
            old_sign = signs[i]
            signs[i] = -old_sign
            
            # Update ap_sums for APs containing i
            for ap_id in pos_to_aps[i]:
                ap_sums[ap_id] -= 2 * old_sign
            
            # Check new max
            new_max = np.max(np.abs(ap_sums))
            
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                # Revert
                signs[i] = old_sign
                for ap_id in pos_to_aps[i]:
                    ap_sums[ap_id] += 2 * old_sign
    
    # Convert back to list of Python integers
    return [int(s) for s in signs]
# EVOLVE-BLOCK-END
