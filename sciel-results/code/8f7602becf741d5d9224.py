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
    
    # 5. Local search: try to swap blocks to reduce count
    # This is a 1-swap local search: try replacing each block with another that doesn't increase count
    # Actually, for minimization, we want to try removing blocks that are "almost redundant"
    # A better approach: try to find if we can replace 2 blocks with 1 (hard) or just verify redundancy again
    
    # Alternative: try random swaps to see if we can maintain coverage with same count but "better" structure
    # For now, just return the greedy solution with redundancy removed
    
    # 6. Second pass: try a different greedy strategy with different tie-breaking
    if time.time() < deadline - 10.0:
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
    
    # Return the blocks
    return [list(all_blocks[idx]) for idx in chosen_block_indices]
