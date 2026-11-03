import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Precompute indices for t-subsets
    # Map each t-subset to an integer index 0..N-1
    t_subsets = list(itertools.combinations(range(v), t))
    N = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Full mask of all t-subsets
    full_mask = (1 << N) - 1
    
    # 2. Precompute coverage masks for all k-blocks
    # k_blocks: list of lists of k points
    k_blocks = list(itertools.combinations(range(v), k))
    
    # For each block, compute the bitset of t-subsets it covers
    block_masks = []
    for block in k_blocks:
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= (1 << subset_to_idx[sub])
        block_masks.append(mask)
        
    # 3. Greedy Construction
    # We want to cover full_mask
    uncovered = full_mask
    selected_blocks = []
    
    # Optimization: Sort blocks by potential? Or just iterate.
    # With N=3876 and ~50k blocks, iterating all blocks each step is ~50k ops.
    # 150 steps * 50k = 7.5M ops. Fast.
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_block_idx = -1
        
        # Find block with max gain
        # gain = popcount(block_mask & uncovered)
        for i, mask in enumerate(block_masks):
            if not mask:
                continue
            newly_covered = mask & uncovered
            # bit_count is fast in Python 3.8+
            gain = newly_covered.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                # Early exit if we found a block that covers all remaining?
                # Unlikely in one step.
        
        if best_block_idx == -1:
            # Should not happen if problem is solvable
            break
            
        selected_blocks.append(best_block_idx)
        uncovered &= ~block_masks[best_block_idx]
        
    # 4. Post-hoc Redundancy Removal
    if selected_blocks:
        # Count how many times each t-subset is covered
        counts = [0] * N
        
        for idx in selected_blocks:
            mask = block_masks[idx]
            # Iterate set bits in mask
            m = mask
            while m:
                # Get index of lowest set bit
                lsb = m & -m
                idx_bit = lsb.bit_length() - 1
                counts[idx_bit] += 1
                m ^= lsb
                
        # Identify unique bits (count == 1)
        unique_mask = 0
        for i in range(N):
            if counts[i] == 1:
                unique_mask |= (1 << i)
                
        # Remove redundant blocks
        changed = True
        while changed:
            changed = False
            new_selected = []
            for idx in selected_blocks:
                mask = block_masks[idx]
                # If this block covers any unique bit, it's essential
                if mask & unique_mask:
                    new_selected.append(idx)
                else:
                    # Redundant: remove it
                    changed = True
                    # Update counts
                    m = mask
                    while m:
                        lsb = m & -m
                        idx_bit = lsb.bit_length() - 1
                        counts[idx_bit] -= 1
                        if counts[idx_bit] == 1:
                            unique_mask |= (1 << idx_bit)
                        elif counts[idx_bit] == 0:
                            unique_mask &= ~(1 << idx_bit)
                        m ^= lsb
            
            selected_blocks = new_selected
            
    # 5. Convert indices back to blocks
    result = [list(k_blocks[i]) for i in selected_blocks]
    
    return result
