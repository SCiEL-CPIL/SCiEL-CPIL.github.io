import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    all_mask = (1 << n_t_subsets) - 1
    
    # 2. Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    n_k_subsets = len(k_subsets)
    
    # 3. Compute bitmask for each k-subset
    masks = [0] * n_k_subsets
    # Also compute the list of t-subset indices for each block (for fast updates)
    block_t_indices = [[] for _ in range(n_k_subsets)]
    
    for i, block in enumerate(k_subsets):
        mask = 0
        indices = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
            indices.append(idx)
        masks[i] = mask
        block_t_indices[i] = indices
        
    # 4. Precompute coverers: for each t-subset, which blocks cover it?
    # coverers[t_idx] = list of block indices
    coverers = [[] for _ in range(n_t_subsets)]
    for i in range(n_k_subsets):
        for t_idx in block_t_indices[i]:
            coverers[t_idx].append(i)
            
    # 5. Greedy Set Cover with "Hardest Element" heuristic
    uncovered = all_mask
    selected_greedy = []
    used_blocks = [False] * n_k_subsets
    
    # avail_count[t_idx] = number of blocks in the REMAINING pool that cover t_idx
    # Initially, all blocks are in the pool.
    avail_count = [len(c) for c in coverers]
    
    # We need to efficiently find the t-subset in 'uncovered' with min avail_count.
    # We can just iterate through all t_subsets? 11628 iterations per step.
    # Steps ~ 400. Total 4.6 million iterations. Fast.
    
    while uncovered:
        # Find hardest t-subset
        min_avail = float('inf')
        hardest_t = -1
        
        # Iterate through set bits of uncovered
        # It's faster to iterate through all t_indices and check if bit is set?
        # Or use bit tricks?
        # 11628 is small enough to just loop.
        for t_idx in range(n_t_subsets):
            if (uncovered >> t_idx) & 1:
                if avail_count[t_idx] < min_avail:
                    min_avail = avail_count[t_idx]
                    hardest_t = t_idx
                    if min_avail == 1:
                        break # Can't get better than 1
        
        if hardest_t == -1:
            break # Should not happen if uncovered is not 0
            
        # Find best block among coverers[hardest_t] that is not used
        best_block = -1
        best_gain = -1
        
        for b_idx in coverers[hardest_t]:
            if used_blocks[b_idx]:
                continue
            gain = (masks[b_idx] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
                
        if best_block == -1:
            break # Dead end
            
        # Select best_block
        selected_greedy.append(best_block)
        used_blocks[best_block] = True
        uncovered &= ~masks[best_block]
        
        # Update avail_count
        # For every t-subset covered by best_block, the number of available coverers decreases by 1
        for t_idx in block_t_indices[best_block]:
            avail_count[t_idx] -= 1
            
    current_best = len(selected_greedy)
    best_solution = selected_greedy[:]
    
    # 6. Fast Redundancy Removal
    # Build coverage counts for the current solution
    def get_reduced_solution(sol):
        # sol is list of block indices
        if not sol:
            return []
        
        # Count how many times each t-subset is covered
        counts = [0] * n_t_subsets
        for b_idx in sol:
            for t_idx in block_t_indices[b_idx]:
                counts[t_idx] += 1
                
        # Try to remove blocks
        # We create a copy to modify
        new_sol = sol[:]
        removed_any = True
        while removed_any:
            removed_any = False
            # Iterate in reverse to handle pops easily? Or just filter.
            # Let's filter.
            new_sol_copy = []
            for b_idx in new_sol:
                # Temporarily remove b_idx
                for t_idx in block_t_indices[b_idx]:
                    counts[t_idx] -= 1
                # Check if any count is 0
                is_redundant = True
                for t_idx in block_t_indices[b_idx]:
                    if counts[t_idx] == 0:
                        is_redundant = False
                        break
                if is_redundant:
                    # Keep it removed
                    removed_any = True
                else:
                    # Restore
                    for t_idx in block_t_indices[b_idx]:
                        counts[t_idx] += 1
                    new_sol_copy.append(b_idx)
            new_sol = new_sol_copy
        return new_sol

    best_solution = get_reduced_solution(best_solution)
    
    # 7. Local Search: 1-Exchange
    # Try to swap a block in solution with a block outside to enable redundancy removal.
    # This is complex and might be slow.
    # Given the time, let's try a few random restarts of the Greedy with slight variations.
    
    start_time = time.time()
    
    # Multi-start Greedy
    # The "Hardest" heuristic is deterministic.
    # We can randomize the tie-breaking in "best_gain".
    
    for _ in range(20):
        if time.time() - start_time > 100:
            break
            
        uncovered = all_mask
        selected = []
        used_blocks = [False] * n_k_subsets
        avail_count = [len(c) for c in coverers]
        
        while uncovered:
            min_avail = float('inf')
            hardest_t = -1
            # Randomize tie-breaking for hardest_t if needed?
            # Just pick the first one with min_avail.
            for t_idx in range(n_t_subsets):
                if (uncovered >> t_idx) & 1:
                    if avail_count[t_idx] < min_avail:
                        min_avail = avail_count[t_idx]
                        hardest_t = t_idx
                        if min_avail == 1:
                            break
            
            if hardest_t == -1:
                break
                
            best_block = -1
            best_gain = -1
            # Add some randomness to gain selection?
            # Pick from top 5 gains?
            gains = []
            for b_idx in coverers[hardest_t]:
                if not used_blocks[b_idx]:
                    gain = (masks[b_idx] & uncovered).bit_count()
                    if gain > 0:
                        gains.append((gain, b_idx))
            
            if not gains:
                break
                
            gains.sort(reverse=True)
            # Pick randomly from top 10 or just best?
            # Let's pick best to be safe, or random from top 5 if time permits.
            # For now, pick best.
            best_block = gains[0][1]
            
            selected.append(best_block)
            used_blocks[best_block] = True
            uncovered &= ~masks[best_block]
            
            for t_idx in block_t_indices[best_block]:
                avail_count[t_idx] -= 1
                
        # Reduce
        reduced = get_reduced_solution(selected)
        
        if len(reduced) < len(best_solution):
            best_solution = reduced
            if len(best_solution) <= 335: # Best known
                break # Can't do better than known lower bound? No, lower bound is 264.
                # But 335 is best known. If we hit it, we stop.
                
    return [list(k_subsets[i]) for i in best_solution]
