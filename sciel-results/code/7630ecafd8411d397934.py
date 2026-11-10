import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. Precompute indices for all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Precompute masks for all k-blocks
    # A block is a k-subset. Its mask has bits set for all t-subsets it contains.
    k_blocks = list(itertools.combinations(range(v), k))
    block_masks = []
    block_indices = []
    
    for block in k_blocks:
        mask = 0
        # For each t-subset within this block, set the corresponding bit
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        block_indices.append(block)
        
    num_blocks = len(k_blocks)
    
    # 3. Greedy Selection
    # uncovered_mask starts with all bits set
    uncovered_mask = (1 << num_t_subsets) - 1
    selected_block_indices = []
    selected_masks = []
    
    # We need to efficiently find the block with max overlap with uncovered_mask
    # Since num_blocks is ~50k and num_t_subsets is ~4k, using Python big ints is fine.
    # We can use a list of (mask, index) and iterate.
    
    # To speed up, we can maintain a list of active blocks.
    # But iterating 50k blocks and doing popcount on 4k-bit ints is fast in Python?
    # 50k * (4000/8) bytes ops... maybe a bit slow in pure Python loop.
    # Let's use NumPy for vectorized popcount if possible, or just optimize the loop.
    # Actually, Python's `int.bit_count()` is very fast (C implementation).
    # 50,000 iterations * ~100ns per bit_count? No, bit_count on 4000 bits is likely < 1us.
    # 50k * 1us = 50ms per step. We need ~150 steps. 150 * 50ms = 7.5s. This is acceptable.
    
    # Optimization: If a block is already "fully covered" by selected blocks, we can skip it?
    # No, we select based on uncovered.
    
    # Let's implement the greedy loop
    while uncovered_mask:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_idx = -1
        
        # Iterate through all possible blocks
        # We can prune: if a block's mask is a subset of the union of selected masks? 
        # No, we just need to find max overlap with uncovered.
        
        for i in range(num_blocks):
            # Calculate overlap
            overlap = block_masks[i] & uncovered_mask
            if overlap:
                gain = overlap.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    # Early exit if we found a perfect cover for all remaining? Unlikely.
                    # If gain == num_t_subsets_remaining, we could break, but we don't track that easily.
        
        if best_idx == -1:
            break
            
        selected_block_indices.append(best_idx)
        selected_masks.append(block_masks[best_idx])
        uncovered_mask &= ~block_masks[best_idx]
        
    # 4. Post-processing: Remove redundant blocks
    # A block is redundant if all its subsets are covered by other selected blocks.
    # We can check this by:
    # union_of_others_mask = OR of all selected masks except this one.
    # If (block_mask & ~union_of_others_mask) == 0, it is redundant.
    
    # This is O(N^2) if done naively. N is ~150. 150^2 = 22500 operations. Fast.
    
    final_indices = []
    for i, idx in enumerate(selected_block_indices):
        if time.time() > deadline:
            final_indices.append(idx)
            continue
            
        # Check if redundant
        redundant = True
        # Calculate union of all OTHER masks
        # Optimization: Precompute prefix/suffix unions? Or just loop.
        # Loop is fine for N=150.
        other_union = 0
        for j, other_idx in enumerate(selected_block_indices):
            if i != j:
                other_union |= selected_masks[j]
                
        # If the block covers any subset NOT covered by others, it's not redundant
        unique_coverage = block_masks[idx] & ~other_union
        if unique_coverage == 0:
            redundant = True
        else:
            redundant = False
            
        if not redundant:
            final_indices.append(idx)
            
    # Convert indices back to blocks
    result_blocks = [list(block_indices[i]) for i in final_indices]
    
    return result_blocks
