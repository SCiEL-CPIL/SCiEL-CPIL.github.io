import itertools
import sys

def construct(v, k, t):
    points = list(range(v))
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Precompute coverage: for each block, list of t-subset indices it covers
    # And bitmask for fast intersection
    block_cov = []
    block_masks = [0] * n_blocks
    
    for b_idx, B in enumerate(all_blocks):
        B_set = set(B)
        cov = []
        mask = 0
        for s_idx, S in enumerate(t_subs):
            # Check if S is subset of B
            # S is a tuple of t points
            # We can check if all points in S are in B_set
            if all(x in B_set for x in S):
                cov.append(s_idx)
                mask |= (1 << s_idx)
        block_cov.append(cov)
        block_masks[b_idx] = mask
        
    # Greedy Phase
    uncovered_mask = (1 << n_t_subs) - 1
    blocks = []
    count = [0] * n_t_subs
    
    # To speed up greedy, we can't just scan all blocks if it's too slow, 
    # but 25M ops is okay.
    while uncovered_mask:
        best_b = -1
        max_gain = -1
        # Scan all blocks
        for b_idx in range(n_blocks):
            gain = (block_masks[b_idx] & uncovered_mask).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_b = b_idx
                
        if best_b == -1:
            break
            
        blocks.append(best_b)
        uncovered_mask &= ~block_masks[best_b]
        
        # Update counts
        for s_idx in block_cov[best_b]:
            count[s_idx] += 1
            
    # Local Search Phase: Remove Redundant Blocks
    # Repeat until no more redundant blocks can be removed
    changed = True
    while changed:
        changed = False
        # Iterate in reverse to safely remove
        for i in range(len(blocks) - 1, -1, -1):
            b = blocks[i]
            is_redundant = True
            for s_idx in block_cov[b]:
                if count[s_idx] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove block b
                for s_idx in block_cov[b]:
                    count[s_idx] -= 1
                blocks.pop(i)
                changed = True
                
    # Convert back to list of lists
    result = [list(all_blocks[b]) for b in blocks]
    return result
