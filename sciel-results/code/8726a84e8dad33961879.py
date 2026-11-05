import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    # Precompute all t-subsets and blocks as bitmasks
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_mask_to_idx = {}
    tsubset_masks = []
    for i, ts in enumerate(all_tsubsets):
        mask = 0
        for p in ts:
            mask |= (1 << p)
        tsubset_masks.append(mask)
        tsubset_mask_to_idx[mask] = i

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_masks = []
    block_covered_ts_indices = []
    
    for blk in all_blocks:
        mask = 0
        for p in blk:
            mask |= (1 << p)
        block_masks.append(mask)
        
        covered_indices = []
        for ts in itertools.combinations(blk, t):
            ts_mask = 0
            for p in ts:
                ts_mask |= (1 << p)
            idx = tsubset_mask_to_idx[ts_mask]
            covered_indices.append(idx)
        block_covered_ts_indices.append(covered_indices)

    # Precompute ts_to_blocks
    ts_to_blocks = [[] for _ in range(n_tsubsets)]
    for b_idx, ts_list in enumerate(block_covered_ts_indices):
        for ts_idx in ts_list:
            ts_to_blocks[ts_idx].append(b_idx)

    # Build coverage matrix: cov_matrix[b_idx, ts_idx] = 1 if block b covers ts
    # Use sparse representation for memory efficiency
    # For vectorized greedy, we need to compute gains quickly
    
    def run_greedy_reduce(seed_val):
        random.seed(seed_val)
        
        counts = np.zeros(n_tsubsets, dtype=np.int8)
        selected_block_indices = []
        remaining_mask = np.ones(n_tsubsets, dtype=bool)
        remaining_count = n_tsubsets
        
        while remaining_count > 0:
            if time.time() > deadline:
                break
            
            # Pick a pivot: choose from remaining with some randomness
            # Weighted by inverse frequency to diversify
            remaining_indices = np.where(remaining_mask)[0]
            if len(remaining_indices) == 0:
                break
            
            # Pick pivot with some randomness - use random selection
            pivot_ts = remaining_indices[random.randint(0, len(remaining_indices)-1)]
            
            # Get candidate blocks covering pivot
            candidates = ts_to_blocks[pivot_ts]
            if not candidates:
                break
            
            # Vectorized gain computation for all candidates
            # For each candidate block, count how many remaining ts it covers
            # Using precomputed block_covered_ts_indices
            
            best_block_idx = -1
            best_gain = -1
            
            # Fast path: if few candidates, do direct
            if len(candidates) < 50:
                for b_idx in candidates:
                    gain = 0
                    for ts_idx in block_covered_ts_indices[b_idx]:
                        if remaining_mask[ts_idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = b_idx
            else:
                # Vectorized: build a matrix of covered ts for candidates
                # This is expensive to build each time, so use a different approach
                # Use numpy: for each candidate, compute sum of remaining_mask[covered_ts]
                cand_arr = np.array(candidates)
                # Get all covered ts for these candidates
                # This requires gathering, which is tricky without scipy sparse
                # Fall back to loop but with early termination
                for b_idx in candidates:
                    gain = 0
                    for ts_idx in block_covered_ts_indices[b_idx]:
                        if remaining_mask[ts_idx]:
                            gain += 1
                            if gain > best_gain:
                                # Early: can't beat if we already have high
                                pass
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
                
            selected_block_indices.append(best_block_idx)
            
            # Update counts and remaining
            for ts_idx in block_covered_ts_indices[best_block_idx]:
                if remaining_mask[ts_idx]:
                    remaining_mask[ts_idx] = False
                    remaining_count -= 1
                counts[ts_idx] += 1
            
        # Reduction Phase
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_block_indices) - 1, -1, -1):
                b_idx = selected_block_indices[i]
                is_redundant = True
                for ts_idx in block_covered_ts_indices[b_idx]:
                    if counts[ts_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    selected_block_indices.pop(i)
                    for ts_idx in block_covered_ts_indices[b_idx]:
                        counts[ts_idx] -= 1
                    changed = True
                    if time.time() > deadline:
                        return selected_block_indices
        
        return selected_block_indices

    # Swap-based local search to improve a solution
    def local_search_swap(block_list, max_iters=500):
        """Try to improve by swapping one block for another that reduces size or enables removal"""
        if not block_list:
            return block_list
        
        # Build coverage counts
        counts = np.zeros(n_tsubsets, dtype=np.int8)
        for b_idx in block_list:
            for ts_idx in block_covered_ts_indices[b_idx]:
                counts[ts_idx] += 1
        
        # Find blocks that are "critical" (cover some ts with count==1)
        # A block can be removed if all its ts have count > 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            
            # Try to remove a redundant block
            for i in range(len(block_list)):
                b_idx = block_list[i]
                can_remove = True
                for ts_idx in block_covered_ts_indices[b_idx]:
                    if counts[ts_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove it
                    block_list.pop(i)
                    for ts_idx in block_covered_ts_indices[b_idx]:
                        counts[ts_idx] -= 1
                    improved = True
                    break
            
            if not improved:
                # Try swap: replace one block with another that allows more removals
                # This is expensive; do limited attempts
                if iters < 50:  # Limit swap attempts
                    random_i = random.randint(0, len(block_list)-1)
                    old_b = block_list[random_i]
                    
                    # Find blocks that cover at least one ts currently covered only by old_b
                    # Actually, try random swaps and check if still valid + smaller
                    for _ in range(20):
                        new_b = random.randint(0, n_blocks-1)
                        if new_b == old_b:
                            continue
                        
                        # Check if replacing old_b with new_b keeps coverage
                        # New counts: decrement old_b's ts, increment new_b's ts
                        # Must ensure no ts drops to 0
                        valid = True
                        for ts_idx in block_covered_ts_indices[old_b]:
                            if counts[ts_idx] == 1:
                                # This ts is only covered by old_b
                                # Check if new_b covers it
                                new_b_covers = False
                                for ts2 in block_covered_ts_indices[new_b]:
                                    if ts2 == ts_idx:
                                        new_b_covers = True
                                        break
                                if not new_b_covers:
                                    valid = False
                                    break
                        
                        if valid:
                            # Perform swap
                            for ts_idx in block_covered_ts_indices[old_b]:
                                counts[ts_idx] -= 1
                            for ts_idx in block_covered_ts_indices[new_b]:
                                counts[ts_idx] += 1
                            block_list[random_i] = new_b
                            improved = True
                            break
        
        return block_list

    best_solution = None
    best_len = float('inf')
    
    num_restarts = 25
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        sol = run_greedy_reduce(i * 12345 + 7)
        
        # Apply local search
        if sol and time.time() < deadline - 10:
            sol = local_search_swap(sol, max_iters=100)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol[:]
    
    # Convert block indices back to lists of points
    final_blocks = []
    if best_solution:
        for b_idx in best_solution:
            final_blocks.append(list(all_blocks[b_idx]))
    else:
        sol = run_greedy_reduce(0)
        for b_idx in sol:
            final_blocks.append(list(all_blocks[b_idx]))

    return final_blocks
