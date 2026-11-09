import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a deterministic greedy set cover approach
    followed by a redundancy removal phase and local search optimization.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage bitmask for every possible k-block
    block_masks = []
    all_blocks = []
    
    # Generate all k-blocks and their masks
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    num_blocks = len(block_masks)
    
    # Convert to numpy array for faster operations
    block_masks_np = np.array(block_masks, dtype=object)
    
    # Helper for bit count
    def count_bits(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')
    
    # 3. Deterministic greedy set cover
    current_uncovered = (1 << num_t_subsets) - 1
    chosen_block_indices = []
    chosen_block_masks = []
    current_union = 0
    
    while current_uncovered != 0:
        if time.time() > deadline - 5.0:
            break
        
        best_gain = -1
        best_block_idx = -1
        best_tie = -1
        
        # Find block with maximum gain, using tie-breaking based on overlap with current union
        for i in range(num_blocks):
            intersection = block_masks[i] & current_uncovered
            gain = count_bits(intersection)
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                # Tie-breaker: prefer blocks that overlap more with current union (consolidation)
                best_tie = count_bits(block_masks[i] & current_union)
            elif gain == best_gain and gain > 0:
                tie = count_bits(block_masks[i] & current_union)
                if tie > best_tie:
                    best_tie = tie
                    best_block_idx = i
        
        if best_block_idx == -1:
            break
        
        chosen_block_indices.append(best_block_idx)
        chosen_block_masks.append(block_masks[best_block_idx])
        current_uncovered &= ~block_masks[best_block_idx]
        current_union |= block_masks[best_block_idx]
    
    # 4. Post-processing: Remove redundant blocks from solution
    n_chosen = len(chosen_block_indices)
    
    changed = True
    while changed:
        changed = False
        for i in range(n_chosen):
            if time.time() > deadline - 3.0:
                break
            
            mask_i = chosen_block_masks[i]
            
            # Calculate union of others
            union_others = 0
            for j in range(n_chosen):
                if i != j:
                    union_others |= chosen_block_masks[j]
            
            # Check if mask_i is subset of union_others
            if (mask_i & ~union_others) == 0:
                # Remove block i
                chosen_block_indices.pop(i)
                chosen_block_masks.pop(i)
                n_chosen -= 1
                changed = True
                break  # Restart loop as indices shifted
        
        if time.time() > deadline - 3.0:
            break
    
    # 5. Local search: 1-exchange - try removing each block and re-covering lost subsets
    # with at most one new block. This can reduce block count.
    def try_remove_and_replace(chosen_indices, chosen_masks):
        """Try to remove one block and replace its unique coverage with at most one new block."""
        n = len(chosen_indices)
        if n < 2:
            return chosen_indices, chosen_masks, False
        
        # Precompute for each block, which t-subsets it uniquely covers
        # For each block i, unique_i = mask_i & ~(union of all other masks)
        # If unique_i is empty, block is redundant (already handled).
        # If unique_i is non-empty, we need to find a single block that covers all of unique_i.
        
        for i in range(n):
            if time.time() > deadline - 3.0:
                break
            
            mask_i = chosen_masks[i]
            
            # Compute union of all other blocks
            union_others = 0
            for j in range(n):
                if i != j:
                    union_others |= chosen_masks[j]
            
            unique_i = mask_i & ~union_others
            if unique_i == 0:
                # Redundant - remove
                chosen_indices.pop(i)
                chosen_masks.pop(i)
                return chosen_indices, chosen_masks, True
            
            # Try to find a single block that covers all of unique_i
            # unique_i must be subset of some block's mask
            best_block_idx = -1
            best_overlap = -1
            
            # unique_i has at most C(7,4)=35 bits set (since block has 7 elements, 4-subsets)
            # We need a block whose mask contains all bits in unique_i
            # i.e., (block_mask & unique_i) == unique_i
            
            # Vectorized check: for each block, check if it covers unique_i
            # (block_masks_np[j] & unique_i) == unique_i
            # This means block_masks_np[j] has 1s wherever unique_i has 1s
            
            # Use numpy: count bits in (block_masks_np[j] & unique_i), must equal count_bits(unique_i)
            unique_count = count_bits(unique_i)
            
            # Check all blocks
            for j in range(num_blocks):
                if chosen_indices and j in chosen_indices:
                    continue
                # Check if block j covers all of unique_i
                if (block_masks[j] & unique_i) == unique_i:
                    # Found a replacement
                    # Prefer block with most overlap with existing union (consolidation)
                    overlap = count_bits(block_masks[j] & union_others)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_block_idx = j
            
            if best_block_idx != -1:
                # Replace block i with best_block_idx
                chosen_indices[i] = best_block_idx
                chosen_masks[i] = block_masks[best_block_idx]
                return chosen_indices, chosen_masks, True
        
        return chosen_indices, chosen_masks, False
    
    # Run 1-exchange local search iteratively until no improvement
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(len(chosen_block_indices)):
            if time.time() > deadline - 3.0:
                break
            new_indices, new_masks, changed = try_remove_and_replace(
                list(chosen_block_indices), list(chosen_block_masks))
            if changed:
                chosen_block_indices = new_indices
                chosen_block_masks = new_masks
                improved = True
                break
    
    # 6. Second pass: try a different greedy strategy with different tie-breaking
    if time.time() < deadline - 15.0:
        # Try greedy with diversity tie-breaking (prefer less overlap)
        current_uncovered2 = (1 << num_t_subsets) - 1
        chosen2 = []
        current_union2 = 0
        
        while current_uncovered2 != 0:
            if time.time() > deadline - 5.0:
                break
            
            best_gain2 = -1
            best_block_idx2 = -1
            best_tie2 = float('inf')
            
            for i in range(num_blocks):
                intersection = block_masks[i] & current_uncovered2
                gain2 = count_bits(intersection)
                
                if gain2 > best_gain2:
                    best_gain2 = gain2
                    best_block_idx2 = i
                    best_tie2 = count_bits(block_masks[i] & current_union2)
                elif gain2 == best_gain2 and gain2 > 0:
                    tie2 = count_bits(block_masks[i] & current_union2)
                    if tie2 < best_tie2:
                        best_tie2 = tie2
                        best_block_idx2 = i
            
            if best_block_idx2 == -1:
                break
            
            chosen2.append(best_block_idx2)
            current_uncovered2 &= ~block_masks[best_block_idx2]
            current_union2 |= block_masks[best_block_idx2]
        
        # Remove redundancies from second solution
        chosen2_masks = [block_masks[idx] for idx in chosen2]
        n2 = len(chosen2)
        changed2 = True
        while changed2:
            changed2 = False
            for i in range(n2):
                if time.time() > deadline - 3.0:
                    break
                mask_i = chosen2_masks[i]
                union_others = 0
                for j in range(n2):
                    if i != j:
                        union_others |= chosen2_masks[j]
                if (mask_i & ~union_others) == 0:
                    chosen2.pop(i)
                    chosen2_masks.pop(i)
                    n2 -= 1
                    changed2 = True
                    break
            if time.time() > deadline - 3.0:
                break
        
        # Use the smaller solution
        if len(chosen2) < len(chosen_block_indices):
            chosen_block_indices = chosen2
            chosen_block_masks = [block_masks[idx] for idx in chosen_block_indices]
    
    # 7. Final local search pass on the best solution
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for i in range(len(chosen_block_indices)):
                if time.time() > deadline - 2.0:
                    break
                new_indices, new_masks, changed = try_remove_and_replace(
                    list(chosen_block_indices), list(chosen_block_masks))
                if changed:
                    chosen_block_indices = new_indices
                    chosen_block_masks = new_masks
                    improved = True
                    break
    
    # Return the blocks
    return [list(all_blocks[idx]) for idx in chosen_block_indices]
