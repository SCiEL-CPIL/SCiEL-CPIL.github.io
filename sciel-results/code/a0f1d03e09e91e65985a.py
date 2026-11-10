import itertools
import time
import random
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit

    # 1. Generate all t-subsets and map them to unique indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (candidate blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 3. Precompute a bitmask for each block using Python integers (arbitrary precision)
    # Bit i is 1 if the i-th t-subset is contained in the block
    block_masks = [0] * num_blocks
    for idx, block in enumerate(all_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            i = t_subset_to_idx[t_sub]
            mask |= (1 << i)
        block_masks[idx] = mask
    
    # 4. Greedy Set Cover Initialization
    # Start with all t-subsets uncovered
    uncovered_mask = (1 << num_t_subsets) - 1
    selected_block_indices = []
    
    while uncovered_mask:
        if time.time() > deadline - 60:
            # If we run out of time for greedy, break. 
            # We rely on local search to fix any gaps if possible, but greedy should be fast enough.
            break
            
        best_gain = -1
        best_idx = -1
        
        # Find the block that covers the most uncovered t-subsets
        for i in range(num_blocks):
            if i in selected_block_indices:
                continue
            gain = (uncovered_mask & block_masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_gain == 0:
            # No improvement possible? This shouldn't happen for valid parameters
            break
            
        selected_block_indices.append(best_idx)
        uncovered_mask &= ~block_masks[best_idx]

    # 5. Local Search Refinement
    # We want to minimize the number of blocks while keeping all t-subsets covered.
    # State: set of selected block indices.
    # Objective: Minimize len(selected_block_indices) subject to coverage constraint.
    
    # Current state
    current_indices = list(selected_block_indices)
    current_set = set(current_indices)
    
    # Compute initial coverage mask (should be full if greedy finished)
    # We can just use a boolean array or bitmask for coverage to check validity quickly
    # Let's use a bitmask for coverage as well for speed, or just check the intersection.
    
    def is_valid(indices):
        """Check if the set of indices covers all t-subsets."""
        cov_mask = 0
        for i in indices:
            cov_mask |= block_masks[i]
        return cov_mask == ((1 << num_t_subsets) - 1)
    
    # Optimize coverage mask computation by maintaining it incrementally
    # For simplicity in the first pass, we'll recompute or use a smart update.
    # Given N=19, K=7, T=4, num_t_subsets = C(19,4) = 3876.
    # Bitmask is ~485 bytes. OR operation is fast.
    
    # Let's use a more efficient local search:
    # 1. Try to remove a block. If coverage remains full, keep removal.
    # 2. Try to swap a block for a better one (though 'better' is hard to define without relaxation).
    #    Actually, just removing redundant blocks is the primary gain.
    #    We can also try to replace a block with another that allows removing *another* block later.
    
    best_indices = current_indices[:]
    best_count = len(best_indices)
    
    # Iterative improvement loop
    # Phase 1: Remove redundant blocks
    # A block is redundant if removing it doesn't uncover any t-subset.
    
    # To speed up redundancy check, we can maintain a coverage count array.
    # coverage_count[ts_idx] = number of blocks in current solution covering ts_idx.
    
    coverage_count = np.zeros(num_t_subsets, dtype=np.int32)
    for bi in current_indices:
        mask = block_masks[bi]
        # Extract set bits from mask and increment coverage_count
        # This is a bit slow in Python loop, but num_blocks is small (~150-200).
        # We can precompute the list of t-subset indices for each block to make this faster.
        
    # Precompute list of t-subset indices for each block for faster coverage update
    block_tsub_indices = [[] for _ in range(num_blocks)]
    for bi in range(num_blocks):
        mask = block_masks[bi]
        indices = []
        # Iterate through set bits
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            indices.append(idx)
            m ^= lsb
        block_tsub_indices[bi] = indices
        
    # Re-initialize coverage count with precomputed indices
    for bi in current_indices:
        for ts_idx in block_tsub_indices[bi]:
            coverage_count[ts_idx] += 1
            
    # Check if initial solution is valid
    if np.any(coverage_count == 0):
        # Greedy didn't finish? Try to finish it quickly or accept invalid?
        # The task says "Valid when every t-subset... lies in at least one block".
        # If invalid, score is 0. We must ensure validity.
        # Let's run a quick fix: add blocks until valid.
        while np.any(coverage_count == 0):
            # Find an uncovered t-subset
            unc_idx = np.where(coverage_count == 0)[0][0]
            # Find a block covering it that minimizes new blocks added (or just any)
            # Actually, just pick a block that covers it.
            # We can look for a block that covers the most uncovered.
            best_b = -1
            best_gain = -1
            for bi in range(num_blocks):
                if bi in current_set:
                    continue
                gain = 0
                for ts_idx in block_tsub_indices[bi]:
                    if coverage_count[ts_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = bi
            if best_b == -1:
                break
            current_indices.append(best_b)
            current_set.add(best_b)
            for ts_idx in block_tsub_indices[best_b]:
                coverage_count[ts_idx] += 1
            
    # Now we have a valid solution.
    # Local Search:
    # Move 1: Remove a random block. If valid, keep. Else revert.
    # Move 2: Swap a random block with a random other block. If valid, keep. (This might not reduce count, but might enable future removals).
    # Move 3: Add a random block? No, we want to minimize.
    
    # Strategy:
    # Repeat until no improvement or time limit:
    #   Try to remove each block one by one (or random order). If valid, remove it.
    #   This is a "deletion heuristic".
    
    # To avoid getting stuck in local minima where a single removal fails but a swap+removal works:
    # We can do:
    # 1. Try removals.
    # 2. If no removals possible, try swaps. A swap is valid if the new set is valid.
    #    But a swap doesn't change the count. It changes the structure.
    #    After a swap, maybe a removal becomes possible.
    
    # Let's implement a loop:
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try to remove blocks
        # Randomize order to avoid bias
        indices_to_try = list(range(len(current_indices)))
        random.shuffle(indices_to_try)
        
        for i in indices_to_try:
            if i >= len(current_indices):
                continue
            bi = current_indices[i]
            
            # Check if removing bi is valid
            # We need to check if any t-subset has coverage count 1 and is covered by bi.
            # If so, removal is invalid.
            can_remove = True
            for ts_idx in block_tsub_indices[bi]:
                if coverage_count[ts_idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove it
                for ts_idx in block_tsub_indices[bi]:
                    coverage_count[ts_idx] -= 1
                current_indices.pop(i)
                current_set.remove(bi)
                improved = True
                if len(current_indices) < best_count:
                    best_count = len(current_indices)
                    best_indices = current_indices[:]
                # Restart the loop? Or continue?
                # Continuing is fine, but the indices shift.
                # It's safer to break and restart the outer loop to handle index shifts cleanly.
                break
        
        if improved:
            continue
            
        # If no removal was possible, try a swap to escape local minimum
        # Pick a random block in the solution
        if len(current_indices) > 0:
            i = random.randrange(len(current_indices))
            old_bi = current_indices[i]
            
            # Pick a random block NOT in the solution
            # To avoid O(N) check, we can just pick random and check set
            new_bi = random.randrange(num_blocks)
            while new_bi in current_set:
                new_bi = random.randrange(num_blocks)
            
            # Check if swapping old_bi with new_bi is valid
            # 1. Remove old_bi
            # 2. Add new_bi
            # 3. Check validity
            
            # Temporary apply removal
            for ts_idx in block_tsub_indices[old_bi]:
                coverage_count[ts_idx] -= 1
            
            # Temporary apply addition
            for ts_idx in block_tsub_indices[new_bi]:
                coverage_count[ts_idx] += 1
            
            # Check validity
            valid_swap = True
            for ts_idx in range(num_t_subsets):
                if coverage_count[ts_idx] == 0:
                    valid_swap = False
                    break
            
            if valid_swap:
                # Keep the swap
                current_indices[i] = new_bi
                current_set.remove(old_bi)
                current_set.add(new_bi)
                improved = True
                # Note: Count didn't change, but structure did.
            else:
                # Revert
                for ts_idx in block_tsub_indices[new_bi]:
                    coverage_count[ts_idx] -= 1
                for ts_idx in block_tsub_indices[old_bi]:
                    coverage_count[ts_idx] += 1

    # Return the best solution found
    # Convert indices back to blocks
    result_blocks = [list(all_blocks[bi]) for bi in best_indices]
    
    # Final validation to be absolutely sure
    # (Optional, but good for robustness)
    final_coverage = np.zeros(num_t_subsets, dtype=np.int32)
    for b in result_blocks:
        for ts in itertools.combinations(b, t):
            final_coverage[t_subset_to_idx[ts]] += 1
            
    if np.any(final_coverage == 0):
        # If somehow invalid, try to fix by adding blocks (should not happen)
        # This is a fallback
        while np.any(final_coverage == 0):
            unc_idx = np.where(final_coverage == 0)[0][0]
            # Find a block covering unc_idx
            found = False
            for bi in range(num_blocks):
                if bi in [all_blocks.index(tuple(b)) for b in result_blocks]: # Slow, but fallback
                    continue
                if (block_masks[bi] >> unc_idx) & 1:
                    result_blocks.append(list(all_blocks[bi]))
                    for ts in itertools.combinations(result_blocks[-1], t):
                        final_coverage[t_subset_to_idx[ts]] += 1
                    found = True
                    break
            if not found:
                break

    return result_blocks
