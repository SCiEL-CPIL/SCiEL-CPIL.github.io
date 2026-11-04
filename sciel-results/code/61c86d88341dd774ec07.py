"""Covering C(19,8,5): optimized randomized minimal cover via iterative removal."""
import numpy as np
import itertools
import random

def construct(v, k, t):
    # Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(k_blocks)
    
    # For each block, compute the list of t-subset indices it covers
    # Also store as numpy arrays for faster indexing if needed, but lists are fine for iteration
    block_t_list = []
    for block in k_blocks:
        covered = [t_index[ts] for ts in itertools.combinations(block, t)]
        block_t_list.append(covered)
        
    # Coverage counts: how many remaining blocks cover each t-subset
    coverage = np.zeros(n_t, dtype=np.int32)
    for covered in block_t_list:
        for idx in covered:
            coverage[idx] += 1
            
    # blocks_for_s[s] = list of block indices that cover t-subset s
    blocks_for_s = [[] for _ in range(n_t)]
    for i in range(n_k):
        for idx in block_t_list[i]:
            blocks_for_s[idx].append(i)
            
    # Convert to numpy arrays for vectorized updates
    blocks_for_s_np = []
    for s in range(n_t):
        if blocks_for_s[s]:
            blocks_for_s_np.append(np.array(blocks_for_s[s], dtype=np.int32))
        else:
            blocks_for_s_np.append(np.array([], dtype=np.int32))

    # Active flags
    active = np.ones(n_k, dtype=bool)
    
    # block_critical_count[i] = number of t-subsets covered by block i that are critical (coverage == 1)
    block_critical_count = np.zeros(n_k, dtype=np.int32)
    
    # Initialize critical counts
    # A t-subset is critical if coverage[s] == 1
    for s in range(n_t):
        if coverage[s] == 1:
            # All active blocks covering s must have their critical count incremented
            # Since all are active initially, just increment all
            block_critical_count[blocks_for_s_np[s]] += 1
            
    # List of removable blocks: those with block_critical_count == 0
    removable_list = []
    for i in range(n_k):
        if block_critical_count[i] == 0:
            removable_list.append(i)
            
    random.shuffle(removable_list)
    
    while removable_list:
        # Pick a random index from the list
        pos = random.randint(0, len(removable_list) - 1)
        i = removable_list[pos]
        
        # Check if it's still removable (stale entry check)
        if block_critical_count[i] != 0:
            # Swap with last and pop to clean up stale entry
            removable_list[pos] = removable_list[-1]
            removable_list.pop()
            continue
            
        # Valid block to remove
        # Remove from list
        removable_list[pos] = removable_list[-1]
        removable_list.pop()
        
        active[i] = False
        
        # Update coverage and critical counts
        for idx in block_t_list[i]:
            old_count = coverage[idx]
            coverage[idx] = old_count - 1
            
            if old_count == 2:
                # Coverage dropped from 2 to 1, so this t-subset is now critical
                # We need to increment block_critical_count for all *other* active blocks covering idx
                # Block i is now inactive, so we exclude it or rely on active mask
                
                # Get all blocks covering idx
                covering_blocks = blocks_for_s_np[idx]
                if len(covering_blocks) > 0:
                    # Mask for active blocks
                    active_mask = active[covering_blocks]
                    active_indices = covering_blocks[active_mask]
                    
                    # Increment critical count for these active blocks
                    block_critical_count[active_indices] += 1
                    # Note: block_critical_count[i] was 0, and i is not in active_indices (active[i]=False),
                    # so we don't need to decrement it or anything. It stays 0, but i is removed from list.
                    # Wait, if i was covering idx, and idx became critical, does i's critical count matter?
                    # No, i is removed. We only care about active blocks.
                    
            # If old_count == 1, this would mean we removed the last cover, which is invalid.
            # But we only remove blocks with critical_count 0.
            # If coverage[idx] was 1, then idx was critical.
            # If idx was critical, any block covering it (including i) would have had critical_count > 0.
            # So i would not have been removable.
            # Thus, we never remove a block covering a critical subset.
            # So coverage[idx] cannot drop from 1 to 0.

    # Collect remaining blocks
    result = [list(k_blocks[i]) for i in range(n_k) if active[i]]
    
    return result
