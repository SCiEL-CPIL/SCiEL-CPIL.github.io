"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Generate candidate vectors: integer vectors with small coordinates
    # Start with vectors of norm squared = 2 (two nonzero ±1 entries)
    # and norm squared = 4 (various patterns)
    
    candidates = []
    
    # Shell m=2: exactly two nonzero entries, each ±1, norm² = 2
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(v)
    
    # Shell m=1: single nonzero entry ±1, norm² = 1
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            candidates.append(v)
    
    # Shell m=4: exactly four nonzero entries, each ±1, norm² = 4
    # This is a large set; generate and use vectorized greedy
    for idx, combo in enumerate(combinations(range(D), 4)):
        if time.time() > deadline - 30:
            break
        for signs in [(1,1,1,1), (1,1,1,-1), (1,1,-1,1), (1,1,-1,-1),
                      (1,-1,1,1), (1,-1,1,-1), (1,-1,-1,1), (1,-1,-1,-1),
                      (-1,1,1,1), (-1,1,1,-1), (-1,1,-1,1), (-1,1,-1,-1),
                      (-1,-1,1,1), (-1,-1,1,-1), (-1,-1,-1,1), (-1,-1,-1,-1)]:
            v = [0] * D
            for pos, s in zip(combo, signs):
                v[pos] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    if n_cand == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    # Compute norms squared
    norms_sq = np.sum(candidates ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Greedy selection with vectorized inner product check
    # We need: for selected vectors u, v: |u-v|² >= max_norm
    # |u-v|² = |u|² + |v|² - 2*u·v >= max_norm_sq
    # So: u·v <= (|u|² + |v|² - max_norm_sq) / 2
    
    selected = []
    selected_indices = []
    
    # Sort candidates by norm squared (prefer smaller norms first for easier constraint)
    # Actually, let's just process in order and use vectorized max inner product tracking
    
    # Precompute: for each candidate, what's the constraint?
    # We'll track the maximum inner product with each selected vector
    
    # Use a different approach: greedily add vectors, maintaining constraint
    # For efficiency, process candidates and check against all selected
    
    # Start with empty set, add vectors one by one if they satisfy constraint
    # To speed up, we can batch check
    
    # Let's use a simpler greedy: iterate through candidates, add if valid
    # But this is O(n * k) which could be slow for large n
    
    # Better: use the insight from knowledge - vectorized greedy with max inner product
    
    # For equal-norm vectors, the constraint simplifies. Let's group by norm.
    
    # Actually, let's just do a straightforward greedy with early termination
    # and use numpy for the inner product checks in batches
    
    selected_vectors = []
    
    # Process candidates in order
    for i in range(n_cand):
        if time.time() > deadline - 10:
            break
        
        v = candidates[i]
        v_norm_sq = norms_sq[i]
        
        if not selected_vectors:
            selected_vectors.append(v)
            selected_indices.append(i)
            continue
        
        # Check if v satisfies constraint with all selected vectors
        # |v - u|² >= max_norm_sq for all u in selected
        # v_norm_sq + u_norm_sq - 2*v·u >= max_norm_sq
        # v·u <= (v_norm_sq + u_norm_sq - max_norm_sq) / 2
        
        # Compute inner products with all selected
        sel_arr = np.array(selected_vectors)
        ips = sel_arr @ v  # inner products
        
        # Constraint: ips[j] <= (v_norm_sq + sel_norms_sq[j] - max_norm_sq) / 2
        sel_norms_sq = np.sum(sel_arr ** 2, axis=1)
        max_ip_allowed = (v_norm_sq + sel_norms_sq - max_norm_sq) / 2.0
        
        if np.all(ips <= max_ip_allowed + 1e-9):
            selected_vectors.append(v)
            selected_indices.append(i)
    
    if not selected_vectors:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    result = np.array(selected_vectors, dtype=np.int64)
    return result
# EVOLVE-BLOCK-END
