import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Generate all t-subsets and map them to IDs
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_id = {sub: i for i, sub in enumerate(t_subs)}
    n_t_subs = len(t_subs)
    
    # 2. Generate all k-blocks and precompute their t-subset ID sets as bitmasks
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # block_t_bitmasks[i] is a bitmask of t-subset IDs contained in block i
    block_t_bitmasks = []
    for block in blocks:
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= (1 << t_sub_to_id[sub])
        block_t_bitmasks.append(mask)
    
    # 3. Greedy Set Cover using bitmasks
    uncovered_mask = (1 << n_t_subs) - 1
    used_blocks = [False] * n_blocks
    result_blocks = []
    
    while uncovered_mask:
        if time.time() > deadline:
            break
        
        best_idx = -1
        best_gain = -1
        
        for i in range(n_blocks):
            if not used_blocks[i]:
                gain = bin(block_t_bitmasks[i] & uncovered_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
        
        if best_gain == 0:
            break
            
        used_blocks[best_idx] = True
        result_blocks.append(list(blocks[best_idx]))
        uncovered_mask &= ~block_t_bitmasks[best_idx]
    
    if time.time() > deadline:
        return result_blocks
    
    # 4. Local search: try to remove blocks that are redundant
    # First, count how many times each t-subset is covered
    coverage_count = [0] * n_t_subs
    for block_idx in result_blocks:
        mask = block_t_bitmasks[blocks.index(tuple(block_idx))]
        # Count bits in mask
        for sub_id in range(n_t_subs):
            if mask & (1 << sub_id):
                coverage_count[sub_id] += 1
    
    # Try removing blocks that don't make any t-subset uncovered
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(result_blocks)):
            if time.time() > deadline:
                break
            
            # Check if removing block i leaves all t-subsets covered
            block = result_blocks[i]
            block_mask = block_t_bitmasks[blocks.index(tuple(block))]
            
            # Count how many t-subsets would become uncovered
            would_uncover = 0
            for sub_id in range(n_t_subs):
                if block_mask & (1 << sub_id):
                    if coverage_count[sub_id] == 1:
                        would_uncover += 1
            
            if would_uncover == 0:
                # Safe to remove
                for sub_id in range(n_t_subs):
                    if block_mask & (1 << sub_id):
                        coverage_count[sub_id] -= 1
                result_blocks.pop(i)
                improved = True
                break
    
    if time.time() > deadline:
        return result_blocks
    
    # 5. Try 2-for-1 swaps: remove two blocks, add one that covers their union
    # This is more expensive, so only do a limited number of attempts
    max_swap_attempts = 50
    swaps_made = 0
    
    for attempt in range(max_swap_attempts):
        if time.time() > deadline:
            break
        
        if len(result_blocks) < 2:
            break
        
        # Try random pairs
        i, j = random.sample(range(len(result_blocks)), 2)
        block_i = result_blocks[i]
        block_j = result_blocks[j]
        
        # Get t-subsets covered by these two blocks
        mask_i = block_t_bitmasks[blocks.index(tuple(block_i))]
        mask_j = block_t_bitmasks[blocks.index(tuple(block_j))]
        union_mask = mask_i | mask_j
        
        # Find a block that covers all t-subsets in union_mask
        # and doesn't introduce too many new t-subsets
        best_new_block = -1
        best_new_mask = 0
        best_excess = float('inf')
        
        for idx in range(n_blocks):
            if used_blocks[idx]:
                continue
            
            mask = block_t_bitmasks[idx]
            # Check if this block covers all of union_mask
            if (mask & union_mask) != union_mask:
                continue
            
            # Count how many new t-subsets this block covers (outside union_mask)
            excess = bin(mask & ~union_mask).count('1')
            if excess < best_excess:
                best_excess = excess
                best_new_block = idx
                best_new_mask = mask
        
        if best_new_block != -1:
            # Perform the swap
            new_block = list(blocks[best_new_block])
            
            # Update coverage counts
            for sub_id in range(n_t_subs):
                if mask_i & (1 << sub_id):
                    coverage_count[sub_id] -= 1
                if mask_j & (1 << sub_id):
                    coverage_count[sub_id] -= 1
                if best_new_mask & (1 << sub_id):
                    coverage_count[sub_id] += 1
            
            # Remove old blocks and add new one
            result_blocks.pop(max(i, j))
            result_blocks.pop(min(i, j))
            result_blocks.append(new_block)
            swaps_made += 1
    
    return result_blocks
