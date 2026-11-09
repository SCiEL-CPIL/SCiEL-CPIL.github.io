import numpy as np
from itertools import combinations

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    t_subs = list(combinations(range(v), t))
    t_sub_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    # Precompute all k-subsets
    k_subs = list(combinations(range(v), k))
    n_k = len(k_subs)
    
    # Build mapping from k-subset to its t-subset indices
    n_t_in_k = len(list(combinations(range(k), t)))
    block_t_indices = np.zeros((n_k, n_t_in_k), dtype=np.int32)
    for i, ks in enumerate(k_subs):
        for j, ts in enumerate(combinations(ks, t)):
            block_t_indices[i, j] = t_sub_idx[ts]
    
    # Uncovered mask
    uncovered = np.ones(n_t, dtype=bool)
    n_uncovered = n_t
    
    # Store indices of selected blocks
    selected_indices = []
    
    # Greedy loop
    while n_uncovered > 0:
        # Compute gain for each block
        gains = uncovered[block_t_indices].sum(axis=1)
        
        # Pick the block with maximum gain
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
        
        selected_indices.append(best_idx)
        
        # Update uncovered mask
        uncovered[block_t_indices[best_idx]] = False
        n_uncovered -= best_gain
    
    # Post-processing: Remove redundant blocks
    # A block is redundant if all its t-subsets are covered by other selected blocks
    counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in selected_indices:
        counts[block_t_indices[b_idx]] += 1
        
    selected_set = set(selected_indices)
    changed = True
    while changed:
        changed = False
        for b_idx in list(selected_set):
            # If every t-subset in this block is covered by at least 2 blocks (including this one),
            # then this block is redundant.
            if np.min(counts[block_t_indices[b_idx]]) > 1:
                selected_set.remove(b_idx)
                counts[block_t_indices[b_idx]] -= 1
                changed = True
                
    # Convert indices to lists of points
    return [list(k_subs[b_idx]) for b_idx in selected_set]
