import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Precompute all k-blocks and their coverage masks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    block_to_mask = [0] * num_blocks
    
    # Precompute frequency of each t-subset (static) for tie-breaking or scoring
    # freq[i] = number of blocks covering t-subset i
    freq = [0] * num_t
    
    for bi, b in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            mask |= (1 << idx)
            freq[idx] += 1
        block_to_mask[bi] = mask
        
    full_uncovered_mask = (1 << num_t) - 1
    
    def greedy_cover(seed=None):
        if seed is not None:
            random.seed(seed)
            
        uncovered = full_uncovered_mask
        selected_blocks = []
        
        # Precompute a list of block indices sorted by some static heuristic?
        # Or just iterate.
        
        while uncovered:
            best_block_idx = -1
            best_gain = 0
            best_score = -1.0
            
            # Iterate through all blocks
            for bi in range(num_blocks):
                gain_mask = uncovered & block_to_mask[bi]
                if gain_mask == 0:
                    continue
                gain = gain_mask.bit_count()
                
                # Score: primarily gain, secondarily rarity (lower freq is better)
                # To avoid the expensive inner loop for rarity, we can use a precomputed 
                # "rarity score" for the block based on static freq, but that's not 
                # dynamic. 
                # A cheap dynamic score: gain * 1.0 + random noise.
                # Or, we can just use gain.
                
                score = float(gain)
                # Add small random noise to break ties and diversify
                score += random.random() * 0.1
                
                if score > best_score:
                    best_score = score
                    best_gain = gain
                    best_block_idx = bi
                    
            if best_block_idx == -1:
                # No block covers any uncovered subset?
                # This should not happen if the covering is possible, 
                # but if we are stuck, break.
                break
                
            selected_blocks.append(best_block_idx)
            uncovered &= ~block_to_mask[best_block_idx]
            
            if time.time() > deadline:
                return None
                
        return selected_blocks

    def prune_blocks(block_indices):
        """Remove redundant blocks."""
        if not block_indices:
            return block_indices
            
        current = list(block_indices)
        if not current:
            return current
            
        # Count coverage of each t-subset by the current set of blocks
        # We only need to know if a block is the *only* one covering a subset.
        # Efficient way:
        # 1. Count how many blocks cover each t-subset in `current`.
        # 2. A block is essential if it covers at least one t-subset with count 1.
        # 3. Remove non-essential blocks.
        # 4. Repeat.
        
        while time.time() < deadline:
            # Count coverage
            count_cover = [0] * num_t
            for bi in current:
                m = block_to_mask[bi]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    count_cover[idx] += 1
                    m ^= lsb
                    
            # Identify essential blocks
            essential = [False] * len(current)
            for i, bi in enumerate(current):
                m = block_to_mask[bi]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if count_cover[idx] == 1:
                        essential[i] = True
                        break
                    m ^= lsb
                    
            old_len = len(current)
            current = [current[i] for i in range(old_len) if essential[i]]
            
            if len(current) < old_len:
                continue
            else:
                break
                
        return current

    # Run multiple restarts
    best_blocks = None
    best_count = float('inf')
    
    # Try a few seeds
    for seed in range(5):
        if time.time() > deadline:
            break
        blocks = greedy_cover(seed=seed)
        if blocks is None:
            continue
        blocks = prune_blocks(blocks)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback: just run greedy once without pruning if time ran out during pruning?
        # Or return empty? No, must return valid.
        # If greedy failed, try again with seed 0 and no pruning?
        blocks = greedy_cover(seed=0)
        if blocks:
            return [all_blocks[bi] for bi in blocks]
        return []
        
    return [all_blocks[bi] for bi in best_blocks]
