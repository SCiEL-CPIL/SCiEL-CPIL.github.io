import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin

    def check_time(budget=5.0):
        return time.time() - start_time > deadline - budget

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Full universe mask: all bits set to 1
    # Note: For v=19, t=5, num_tsubs = 11628. 
    # A Python int can hold this. Bit operations are fast.
    full_mask = (1 << num_tsubs) - 1

    # Precompute all blocks and their corresponding t-subset masks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_masks[i] is an integer where bit j is 1 if t-subset j is in block i
    block_masks = [0] * num_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = tsub_index[ts]
            mask |= (1 << idx)
        block_masks[i] = mask

    # Helper to check if a set of block indices covers everything
    def is_valid_cover(indices):
        union_mask = 0
        for idx in indices:
            union_mask |= block_masks[idx]
        return (union_mask == full_mask)

    # Helper to get the missing mask for a set of indices
    def get_missing_mask(indices):
        union_mask = 0
        for idx in indices:
            union_mask |= block_masks[idx]
        return full_mask & ~union_mask

    # 1. Greedy Initialization to get a valid cover
    def greedy_cover():
        uncovered = full_mask
        selected_indices = []
        remaining_blocks = set(range(num_blocks))
        
        while uncovered:
            if check_time(10.0):
                break
            best_gain = -1
            best_block = -1
            
            for i in remaining_blocks:
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = i
                elif gain == best_gain and best_block != -1:
                    # Random tie-breaker
                    if random.random() < 0.1:
                        best_block = i
                        
            if best_block == -1:
                break
                
            selected_indices.append(best_block)
            remaining_blocks.remove(best_block)
            uncovered &= ~block_masks[best_block]
            
        return selected_indices

    # 2. Local Search Removal (1-Opt) to clean up initial greedy result
    def local_search_remove(indices):
        indices = list(indices)
        if not indices:
            return indices
            
        changed = True
        while changed and not check_time(5.0):
            changed = False
            new_indices = []
            for i, idx in enumerate(indices):
                others_union = 0
                for j, other_idx in enumerate(indices):
                    if j != i:
                        others_union |= block_masks[other_idx]
                        
                if (block_masks[idx] & ~others_union) == 0:
                    changed = True
                else:
                    new_indices.append(idx)
                    
            indices = new_indices
            
        return indices

    # Initial Cover
    initial = greedy_cover()
    if not initial:
        return []
        
    current_indices = local_search_remove(initial)
    best_indices = list(current_indices)
    best_len = len(best_indices)
    
    # 3. Simulated Annealing with Swap Moves
    # State: list of block indices
    # Move: 
    #   1. Try to remove a random block. 
    #      - If valid, accept (size -1).
    #      - If invalid, try to swap with a random other block.
    #        - If swap restores validity, accept (size 0 change, but different composition).
    #   2. Cooling schedule.
    
    def simulated_annealing(indices, max_iters=20000):
        indices = list(indices)
        current_len = len(indices)
        best_len = current_len
        best_indices = list(indices)
        
        T = 5.0
        alpha = 0.999
        iters = 0
        
        # Precompute list of all block indices for random sampling
        all_block_indices = list(range(num_blocks))
        
        while T > 0.01 and iters < max_iters and not check_time(10.0):
            iters += 1
            
            if len(indices) < 2:
                break
            
            # Pick a random block to potentially remove
            i_remove = random.randint(0, len(indices) - 1)
            block_to_remove = indices[i_remove]
            
            # Temporarily remove it
            new_indices = [x for x in indices if x != block_to_remove]
            missing = get_missing_mask(new_indices)
            
            if missing == 0:
                # Removal succeeded! Improvement.
                indices = new_indices
                current_len -= 1
                if current_len < best_len:
                    best_len = current_len
                    best_indices = list(indices)
                # Restart cooling? Or just continue. Usually continue.
                # We might want to cool down slightly less aggressively if we found an improvement
                # but standard SA just continues.
            else:
                # Coverage broken. Try to repair by swapping block_to_remove with a new block.
                # We need a block that covers 'missing'.
                # Sample a few candidates to find one that covers all missing parts.
                # If found, we swap. Size remains same.
                # This is a "neutral" move that changes the landscape.
                
                # To be efficient, we don't check all blocks. We sample.
                # If missing is small, it's easier to find.
                # If missing is large, it's harder.
                
                # Strategy: Sample 50 random blocks. Check if they cover 'missing'.
                found_swap = False
                for _ in range(50):
                    cand_idx = random.choice(all_block_indices)
                    if cand_idx == block_to_remove:
                        continue
                    # Check if cand covers all missing
                    if (missing & ~block_masks[cand_idx]) == 0:
                        # Swap successful
                        # Replace block_to_remove with cand_idx in indices
                        # Since we already removed it from new_indices, just add cand
                        indices = new_indices + [cand_idx]
                        found_swap = True
                        break
                
                if not found_swap:
                    # No swap found. Reject move.
                    # indices remains unchanged (we used new_indices locally)
                    pass
            
            T *= alpha
            
        return best_indices, best_len

    # Run SA with multiple restarts from the cleaned greedy solution
    # We can also perturb the starting solution for restarts
    
    for restart in range(5):
        if check_time(15.0):
            break
            
        # For restarts > 0, perturb the best solution
        if restart == 0:
            start_sa = list(best_indices)
        else:
            # Remove 3 random blocks and repair greedily? 
            # Or just use best_indices.
            start_sa = list(best_indices)
            
        sa_result, sa_len = simulated_annealing(start_sa)
        
        if sa_len < best_len:
            best_len = sa_len
            best_indices = sa_result
            # Update best to use as start for next restart? 
            # Yes, keep the best found.
            
        # Final cleanup after SA
        best_indices = local_search_remove(best_indices)
        if len(best_indices) < best_len:
            best_len = len(best_indices)

    # Convert indices back to blocks
    return [list(all_blocks[i]) for i in best_indices]

# To ensure the function is available at module level as required
# The task says "Keep the function names and signatures of the seed program."
# The seed program defines construct(v, k, t).
# I have defined construct above.
