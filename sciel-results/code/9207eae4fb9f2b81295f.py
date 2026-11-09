import numpy as np
from itertools import combinations
from functools import reduce

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    t_subs = list(combinations(range(v), t))
    t_sub_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    # Precompute all k-subsets
    k_subs = list(combinations(range(v), k))
    n_k = len(k_subs)
    
    # Build mapping from k-subset to its t-subset indices
    # For each k-subset, find the indices of all its t-subsets
    block_t_indices = np.zeros((n_k, len(list(combinations(range(k), t)))), dtype=np.int32)
    for i, ks in enumerate(k_subs):
        for j, ts in enumerate(combinations(ks, t)):
            block_t_indices[i, j] = t_sub_idx[ts]
    
    # Uncovered mask
    uncovered = np.ones(n_t, dtype=bool)
    n_uncovered = n_t
    blocks = []
    
    # Greedy loop
    while n_uncovered > 0:
        # Compute gain for each block: number of currently uncovered t-subsets it covers
        # block_t_indices has shape (n_k, C(k,t))
        # uncovered[block_t_indices] has shape (n_k, C(k,t))
        # sum along axis 1 gives gain for each block
        gains = uncovered[block_t_indices].sum(axis=1)
        
        # Pick the block with maximum gain
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            # Should not happen if there are still uncovered t-subsets
            # But break to avoid infinite loop
            break
        
        blocks.append(k_subs[best_idx])
        
        # Update uncovered mask
        uncovered[block_t_indices[best_idx]] = False
        n_uncovered -= best_gain
    
    # Convert tuples to lists
    return [list(b) for b in blocks]
