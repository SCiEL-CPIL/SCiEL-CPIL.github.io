import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    n_t_subsets = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Precompute for each block, the indices of the t-subsets it covers
    # block_t_indices[i] is an array of indices into all_t_subsets
    block_t_indices = np.zeros((n_blocks, C(k, t)), dtype=np.int32)
    # Need to calculate C(k,t)
    from math import comb
    k_t = comb(k, t)
    block_t_indices = np.zeros((n_blocks, k_t), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        t_subs_in_block = itertools.combinations(block, t)
        for j, ts in enumerate(t_subs_in_block):
            block_t_indices[i, j] = t_subset_to_idx[ts]
            
    best_solution = None
    best_score = float('inf')
    
    # Run multiple iterations of greedy + local search
    while time.time() < deadline:
        # Greedy Construction
        uncovered = np.ones(n_t_subsets, dtype=np.int8)
        selected_blocks = []
        
        while uncovered.sum() > 0:
            # Calculate gains for all blocks
            # gains[i] = number of uncovered t-subsets in block i
            gains = uncovered[block_t_indices].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                # Should not happen if uncovered is not empty, but safety break
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one candidate
            idx = np.random.choice(candidates)
            selected_blocks.append(idx)
            
            # Update uncovered
            uncovered[block_t_indices[idx]] = 0
            
        # Local Optimization: Remove redundant blocks
        # Count coverage for each t-subset
        coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
        for idx in selected_blocks:
            coverage_count[block_t_indices[idx]] += 1
            
        # Try to remove blocks
        # We iterate through the list. If a block is redundant, remove it.
        # We must be careful: removing one block might make another non-redundant.
        # So we loop until no more blocks can be removed.
        changed = True
        while changed:
            changed = False
            new_selected = []
            for idx in selected_blocks:
                # Tentatively remove idx
                coverage_count[block_t_indices[idx]] -= 1
                # Check if any t-subset has count 0
                if np.min(coverage_count) >= 1:
                    # Block idx is redundant, do not add back to new_selected
                    changed = True
                else:
                    # Block idx is needed, add to new_selected and restore count
                    new_selected.append(idx)
                    coverage_count[block_t_indices[idx]] += 1
            selected_blocks = new_selected
            
        # Update best
        if len(selected_blocks) < best_score:
            best_score = len(selected_blocks)
            best_solution = [list(all_blocks[i]) for i in selected_blocks]
            
        # Check time again
        if time.time() > deadline:
            break
            
    # If no solution found (should not happen), return empty or fail
    if best_solution is None:
        # Fallback: return empty list, but this is invalid. 
        # The greedy should always find a solution.
        return []
        
    return best_solution

def C(n, r):
    if r < 0 or r > n:
        return 0
    if r == 0 or r == n:
        return 1
    r = min(r, n - r)
    numerator = 1
    denominator = 1
    for i in range(1, r + 1):
        numerator *= n - r + i
        denominator *= i
    return numerator // denominator
