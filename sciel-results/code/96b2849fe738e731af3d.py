import itertools
import time
import numpy as np

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets
    k_subsets = list(itertools.combinations(points, k))
    n_blocks = len(k_subsets)
    
    # 3. Compute bitmask for each k-subset
    # We store masks as a list of Python ints for fast bitwise ops in local search
    # but we can use NumPy for initial greedy gain calculation if we convert to arrays?
    # Actually, Python int bit_count is very fast. The loop is the issue.
    
    masks = [0] * n_blocks
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        masks[i] = mask
        
    all_mask = (1 << n_t_subsets) - 1
    
    # Greedy Set Cover
    # We want to pick blocks that maximize gain.
    # Current code loops through all blocks every step.
    # With n_blocks ~ 75k and ~300 steps, that's 22M bit_count ops.
    # Python bit_count is fast, but the loop overhead is significant.
    
    # Optimization: Maintain a list of active blocks.
    # But even better: NumPy doesn't support arbitrary precision integers easily.
    # So we stick to Python ints but optimize the selection.
    
    uncovered = all_mask
    selected = []
    used = [False] * n_blocks
    
    # Pre-compute block sizes? No, all k.
    
    start_time = time.time()
    
    while uncovered and (time.time() - start_time) < 10: # 10s for greedy
        best_gain = -1
        best_idx = -1
        
        # Iterate through blocks
        # We can skip blocks that are already used
        for i in range(n_blocks):
            if used[i]:
                continue
            gain = (masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == uncovered.bit_count():
                    break # Can't do better than covering all remaining
            
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        used[best_idx] = True
        uncovered &= ~masks[best_idx]
        
    # Local Search to remove redundants
    # This is the most critical part for improvement.
    # Current `try_remove_redundant` is O(N^2).
    # We can do it in O(N * N/word_size) if we use bitsets for "coverage count"?
    # Or simply: For each block, check if it's redundant.
    # To check if block i is redundant:
    # Union of all other blocks must cover all_mask.
    # Equivalent: Is there any t-subset covered ONLY by block i?
    # If yes, not redundant.
    # If no, redundant.
    
    # Efficient check:
    # Maintain a count array `cover_count` of size n_t_subsets.
    # cover_count[j] = number of selected blocks covering t-subset j.
    
    def is_redundant(block_idx, cover_count):
        # Check if any t-subset in masks[block_idx] has count 1
        # We can iterate through the set bits of masks[block_idx]
        m = masks[block_idx]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if cover_count[idx] == 1:
                return False
            m ^= lsb
        return True

    # Build cover_count
    cover_count = [0] * n_t_subsets
    for b in selected:
        m = masks[b]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb

    # Remove redundants
    changed = True
    while changed:
        changed = False
        # Iterate backwards to pop
        for i in range(len(selected) - 1, -1, -1):
            if is_redundant(selected[i], cover_count):
                # Remove block i
                b = selected[i]
                m = masks[b]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    m ^= lsb
                selected.pop(i)
                changed = True
                break # Restart loop as indices changed

    # Try to swap blocks to reduce count?
    # If we remove block A, and add block B, and the new set is still a cover, count is same.
    # To reduce count, we need to remove A and NOT add B, which means A was redundant (handled above).
    # Or remove A and B, and add C? (2->1 swap).
    # This is harder.
    
    # Let's try a simple 1-swap: 
    # Can we replace a block A with a block B such that the number of blocks decreases?
    # No, 1-for-1 keeps count same.
    # The only way to decrease count is to find a set of blocks that can be removed.
    # The redundancy check handles removing 1 block if it's covered by others.
    
    # To go lower, we need to find "essential" blocks that can be "merged".
    # This is generally hard.
    
    # Given the time limit, the redundancy removal is the biggest gain.
    # The current program's redundancy removal was O(N^2) with a full union check every time, which is very slow.
    # My `cover_count` approach is O(N * C(k,t)) which is much faster.
    
    return [list(k_subsets[i]) for i in selected]
