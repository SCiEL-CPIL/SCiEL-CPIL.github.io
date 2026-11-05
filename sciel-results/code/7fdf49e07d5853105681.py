import itertools
import random
import time

def construct(v, k, t):
    points = list(range(v))
    all_subsets = list(itertools.combinations(points, t))
    n_subsets = len(all_subsets)
    subset_to_idx = {sub: i for i, sub in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Precompute bitmask for each block
    # Python integers can handle bitmasks of size n_subsets
    block_masks = [0] * n_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    full_mask = (1 << n_subsets) - 1
    
    def popcount(x):
        return bin(x).count("1")
        
    def greedy_cover(seed):
        random.seed(seed)
        uncovered = full_mask
        selected_indices = []
        
        # To speed up, we can maintain a list of active blocks, 
        # but for v=19, n_blocks=50k, iterating all each time is ~50k * 150 = 7.5M ops per run.
        # With 100 runs, 750M ops. Might be too slow in pure Python.
        # Optimization: Only consider blocks that cover at least one uncovered subset.
        
        while uncovered:
            best_block_idx = -1
            best_gain = 0
            best_tie = float('inf')
            
            # Iterate through all blocks
            for i in range(n_blocks):
                m = block_masks[i]
                if m == 0:
                    continue
                # Quick check: if no intersection, skip
                if (uncovered & m) == 0:
                    continue
                
                gain = popcount(uncovered & m)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    best_tie = random.random()
                elif gain == best_gain and gain > 0:
                    r = random.random()
                    if r < best_tie:
                        best_tie = r
                        best_block_idx = i
                        
            if best_block_idx == -1:
                break
                
            selected_indices.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            
        # Post-processing: Remove redundant blocks
        if not selected_indices:
            return []
            
        # Calculate coverage count for each subset
        # This is O(n_selected * C(k,t))
        coverage_count = [0] * n_subsets
        for idx in selected_indices:
            m = block_masks[idx]
            # Iterate set bits
            temp = m
            while temp:
                lsb = temp & -temp
                idx_sub = lsb.bit_length() - 1
                coverage_count[idx_sub] += 1
                temp ^= lsb
                
        to_keep = set(selected_indices)
        changed = True
        while changed:
            changed = False
            current_blocks = list(to_keep)
            random.shuffle(current_blocks)
            for b_idx in current_blocks:
                if b_idx not in to_keep:
                    continue
                is_redundant = True
                m = block_masks[b_idx]
                temp = m
                while temp:
                    lsb = temp & -temp
                    idx_sub = lsb.bit_length() - 1
                    if coverage_count[idx_sub] == 1:
                        is_redundant = False
                        break
                    temp ^= lsb
                    
                if is_redundant:
                    to_keep.remove(b_idx)
                    m = block_masks[b_idx]
                    temp = m
                    while temp:
                        lsb = temp & -temp
                        idx_sub = lsb.bit_length() - 1
                        coverage_count[idx_sub] -= 1
                        temp ^= lsb
                    changed = True
                    
        return [list(all_blocks[i]) for i in to_keep]

    best_len = float('inf')
    best_blocks = []
    
    # Run multiple restarts
    # Estimate time: 1 run ~ 0.5-1.0s? 
    # If 1s/run, we can do ~100 runs in 120s.
    # Let's try 50 runs first to be safe, or dynamic based on time.
    start_time = time.time()
    num_restarts = 0
    max_restarts = 100
    
    while num_restarts < max_restarts:
        seed = num_restarts
        blocks = greedy_cover(seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        num_restarts += 1
        
        if time.time() - start_time > 110:
            break

    return best_blocks
