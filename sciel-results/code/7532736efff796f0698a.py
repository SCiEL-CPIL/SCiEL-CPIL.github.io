import itertools
import numpy as np
from scipy.sparse import csr_matrix
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # 2. Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(k_blocks)
    
    # 3. Build sparse coverage matrix
    # M[i, j] = 1 if block i covers t-subset j
    rows = []
    cols = []
    for b_idx, block in enumerate(k_blocks):
        # Find all t-subsets of this block
        for t_sub in itertools.combinations(block, t):
            s_idx = subset_to_idx[t_sub]
            rows.append(b_idx)
            cols.append(s_idx)
            
    data = np.ones(len(rows), dtype=np.int8)
    M = csr_matrix((data, (rows, cols)), shape=(n_k, n_t))
    
    # 4. Greedy construction function
    def greedy_construct(seed):
        rng = np.random.default_rng(seed)
        uncovered = np.ones(n_t, dtype=np.int8)
        blocks_used = []
        
        while np.any(uncovered):
            # Calculate gain for each block
            # M @ uncovered gives count of uncovered subsets covered by each block
            gains = M @ uncovered
            
            # Find max gain
            max_gain = np.max(gains)
            
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly to diversify
            if len(candidates) > 1:
                best_block_idx = candidates[rng.integers(len(candidates))]
            else:
                best_block_idx = candidates[0]
                
            blocks_used.append(best_block_idx)
            
            # Update uncovered
            # Get indices of t-subsets covered by this block
            block_row = M.getrow(best_block_idx)
            uncovered[block_row.indices] = 0
            
            # Check time
            if time.time() > deadline:
                break
        
        return [k_blocks[i] for i in blocks_used]
    
    # 5. Pruning function
    def prune_covering(blocks):
        if not blocks:
            return blocks
            
        # Count how many times each t-subset is covered
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            for t_sub in itertools.combinations(b, t):
                s_idx = subset_to_idx[t_sub]
                cover_count[s_idx] += 1
                
        pruned = []
        changed = True
        while changed:
            changed = False
            new_pruned = []
            for b in pruned if pruned else blocks:
                # Check if b is redundant
                redundant = True
                for t_sub in itertools.combinations(b, t):
                    s_idx = subset_to_idx[t_sub]
                    if cover_count[s_idx] == 1:
                        redundant = False
                        break
                if not redundant:
                    # Keep b
                    new_pruned.append(b)
                else:
                    # Remove b, decrement counts
                    for t_sub in itertools.combinations(b, t):
                        s_idx = subset_to_idx[t_sub]
                        cover_count[s_idx] -= 1
                    changed = True
            
            blocks = new_pruned
            pruned = blocks # This logic is slightly off, let's simplify
            
        # Simpler pruning:
        # Just do one pass or few passes
        # Let's restart pruning logic properly
        
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            for t_sub in itertools.combinations(b, t):
                s_idx = subset_to_idx[t_sub]
                cover_count[s_idx] += 1
        
        pruned_blocks = []
        for b in blocks:
            is_redundant = True
            for t_sub in itertools.combinations(b, t):
                s_idx = subset_to_idx[t_sub]
                if cover_count[s_idx] == 1:
                    is_redundant = False
                    break
            if not is_redundant:
                pruned_blocks.append(b)
            else:
                for t_sub in itertools.combinations(b, t):
                    s_idx = subset_to_idx[t_sub]
                    cover_count[s_idx] -= 1
                    
        # Repeat a few times
        for _ in range(3):
            new_pruned = []
            for b in pruned_blocks:
                is_redundant = True
                for t_sub in itertools.combinations(b, t):
                    s_idx = subset_to_idx[t_sub]
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_pruned.append(b)
                else:
                    for t_sub in itertools.combinations(b, t):
                        s_idx = subset_to_idx[t_sub]
                        cover_count[s_idx] -= 1
            pruned_blocks = new_pruned
            
        return pruned_blocks

    best_covering = None
    best_size = float('inf')
    
    # Run multiple greedy restarts
    # Estimate time per run.
    # If n_k is large, M @ uncovered is fast.
    # Let's try 10 restarts first.
    n_restarts = 20
    
    for i in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
            
        covering = greedy_construct(i)
        covering = prune_covering(covering)
        
        if len(covering) < best_size:
            best_size = len(covering)
            best_covering = covering
            
        # Check if we found the lower bound? No, lower bound is 264.
        # If we get close to known best 335, maybe stop? No, try to beat it.
        
    if best_covering is None:
        # Fallback to single greedy
        best_covering = greedy_construct(0)
        best_covering = prune_covering(best_covering)
        
    return best_covering
