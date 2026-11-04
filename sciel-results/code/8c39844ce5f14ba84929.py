import itertools
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Precomputation
    # Generate all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    index_to_comb = all_tsubsets
    comb_to_index = {c: i for i, c in enumerate(all_tsubsets)}
    
    # We need subset_mask[S] for point masks S with popcount(S) <= k
    # subset_mask[S] is an integer where bit i is set if all_tsubsets[i] is a subset of S
    
    # To save memory and time, we only compute for S with popcount <= k.
    # We store in a list indexed by S.
    subset_mask = [0] * (1 << v)
    
    # Generate all masks S with popcount in [t, k]
    valid_masks = []
    for size in range(t, k + 1):
        for comb in itertools.combinations(range(v), size):
            mask = 0
            for p in comb:
                mask |= (1 << p)
            valid_masks.append(mask)
            
    # Compute subset_mask for these masks
    for S in valid_masks:
        # Get points in S
        points = []
        for p in range(v):
            if S & (1 << p):
                points.append(p)
        
        m = 0
        # Iterate t-subsets of points
        for tc in itertools.combinations(points, t):
            idx = comb_to_index[tc]
            m |= (1 << idx)
        subset_mask[S] = m
        
    # 2. Greedy Construction
    uncovered = (1 << n_tsubsets) - 1
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Find the first uncovered t-subset (lexicographically smallest index)
        # Least significant set bit
        lsb = uncovered & -uncovered
        idx = lsb.bit_length() - 1
        
        start_comb = index_to_comb[idx]
        
        # Initialize block with start_comb
        block_points = list(start_comb)
        block_mask = 0
        for p in block_points:
            block_mask |= (1 << p)
            
        # Greedily add points until size k
        while len(block_points) < k:
            best_gain = -1
            best_x = -1
            
            for x in range(v):
                if block_mask & (1 << x):
                    continue
                    
                new_block_mask = block_mask | (1 << x)
                
                # Get masks
                new_t_mask = subset_mask[new_block_mask]
                old_t_mask = subset_mask[block_mask]
                
                # Gain: subsets covered by new_block_mask but not old_block_mask, 
                # that are currently uncovered
                gain_mask = (new_t_mask & ~old_t_mask) & uncovered
                gain = gain_mask.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    
            if best_x == -1:
                 # Fallback: pick any point not in block
                 for x in range(v):
                     if not (block_mask & (1 << x)):
                         best_x = x
                         break
                         
            block_points.append(best_x)
            block_mask |= (1 << best_x)
            
        block_points.sort()
        blocks.append(block_points)
        
        # Update uncovered
        final_mask = subset_mask[block_mask]
        uncovered &= ~final_mask
        
    # 3. Minimization Phase (Structural Change)
    # Convert blocks to masks for faster lookup
    block_masks = [0] * len(blocks)
    for i, b in enumerate(blocks):
        m = 0
        for p in b:
            m |= (1 << p)
        block_masks[i] = m
    
    # Precompute list of t-subset indices covered by each block
    # subset_indices[block_idx] = list of indices into all_tsubsets
    block_t_indices = []
    for bm in block_masks:
        points = []
        for p in range(v):
            if bm & (1 << p):
                points.append(p)
        indices = []
        for tc in itertools.combinations(points, t):
            indices.append(comb_to_index[tc])
        block_t_indices.append(indices)
        
    # Count coverage
    cover_count = [0] * n_tsubsets
    for indices in block_t_indices:
        for idx in indices:
            cover_count[idx] += 1
            
    # Iteratively remove redundant blocks
    # A block is redundant if all its t-subsets are covered by at least one other block
    # i.e., cover_count[s] > 1 for all s in block
    active_blocks = [True] * len(blocks)
    
    # We need to be careful with time. 
    # Max iterations: len(blocks)
    # Cost per iteration: O(M * C(k,t))
    
    max_iter = len(blocks)
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        removed_any = False
        for i in range(len(blocks)):
            if not active_blocks[i]:
                continue
            
            is_redundant = True
            for s_idx in block_t_indices[i]:
                if cover_count[s_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block i
                active_blocks[i] = False
                removed_any = True
                for s_idx in block_t_indices[i]:
                    cover_count[s_idx] -= 1
                    
        if not removed_any:
            break
            
    # Filter blocks
    final_blocks = [blocks[i] for i in range(len(blocks)) if active_blocks[i]]
    
    return final_blocks
