import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin for safety
    
    # 1. Precompute t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # 2. Precompute k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # 3. Precompute block_t_indices: (n_k, C(k, t))
    # C(k, t) is the number of t-subsets in a k-set
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)
    
    # Initialize array with -1 to detect errors, though we expect full coverage
    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    
    # Efficiently fill block_t_indices
    # We iterate over k_subsets and for each, generate its t-subsets
    # Since n_k is ~75k and n_t_in_k is 56, this is ~4.2M lookups.
    # Python loop might be slow, but NumPy indexing is fast if we can vectorize the lookup.
    # However, the lookup depends on the specific values in the k-subset.
    # We can do this in a loop, but let's see if we can speed it up.
    
    # For each block, we need to map its local t-combinations to global t-subset indices.
    # local_t_combo is a tuple of indices 0..k-1.
    # block is a tuple of points p_0..p_{k-1}.
    # The t-subset is (block[i_0], ..., block[i_{t-1}]).
    
    # We can precompute the list of local t-combinations once.
    local_t_combos = t_in_k
    
    # Filling the array
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(local_t_combos):
            # Extract the points
            # tuple(block[p] for p in lc)
            # Lookup in dict
            block_t_indices[i, j] = t_sub_to_idx[tuple(block[p] for p in lc)]
            
    # 4. Greedy Set Cover with Multiple Restarts
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Reset state
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []
        
        # Greedy loop
        while True:
            if time.time() > deadline - 5.0:
                break
            
            # Calculate gains: number of uncovered t-subsets covered by each block
            # uncovered is bool array of size n_t
            # block_t_indices is int array of size (n_k, n_t_in_k)
            # uncovered[block_t_indices] creates a bool array of size (n_k, n_t_in_k)
            # .sum(axis=1) sums along the t-subsets dimension, giving size (n_k,)
            gains = uncovered[block_t_indices].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly to encourage diversity
            best_idx = rng.choice(candidates)
            
            selected_blocks.append(best_idx)
            
            # Update uncovered
            # uncovered[block_t_indices[best_idx]] = False
            # This is a vectorized operation
            uncovered[block_t_indices[best_idx]] = False
            
        # Check if we finished within time limit
        if time.time() > deadline - 5.0:
            break
            
        # Post-optimization: Remove redundant blocks
        if selected_blocks:
            # Build coverage count
            coverage_count = np.zeros(n_t, dtype=np.int32)
            for bi in selected_blocks:
                coverage_count[block_t_indices[bi]] += 1
                
            remaining = list(selected_blocks)
            changed = True
            while changed:
                if time.time() > deadline - 2.0:
                    break
                changed = False
                for i in range(len(remaining)):
                    bi = remaining[i]
                    ts = block_t_indices[bi]
                    # If all t-subsets in this block are covered by > 1 block, it's redundant
                    if np.all(coverage_count[ts] > 1):
                        coverage_count[ts] -= 1
                        remaining.pop(i)
                        changed = True
                        break # Restart the scan as indices changed
                        
            if len(remaining) < best_size:
                best_size = len(remaining)
                best_solution = [k_subsets[i] for i in remaining]
                
    if best_solution is None:
        # Fallback: return empty or a trivial covering if time ran out immediately
        # But we should have at least one restart
        return []
        
    return [list(b) for b in best_solution]
