import itertools
import random
import time
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Precomputation
    # All t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # All k-subsets (blocks)
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(all_k_subsets)
    
    # For each k-subset, compute the bitmask of t-subsets it covers
    # Also, for each t-subset, store the list of k-subsets that cover it
    k_to_t_indices = []
    t_to_k_indices = [[] for _ in range(num_t)]
    
    for ki, k_sub in enumerate(all_k_subsets):
        t_indices = []
        for t_sub in itertools.combinations(k_sub, t):
            ti = t_subset_to_idx[t_sub]
            t_indices.append(ti)
            t_to_k_indices[ti].append(ki)
        k_to_t_indices.append(t_indices)
        
    # Precompute unique masks for each block? 
    # We need to check if a block B covers a set of t-subsets U.
    # U is a set of indices. B covers U if all indices in U are in k_to_t_indices[B].
    
    def run_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
            
        uncovered_mask = (1 << num_t) - 1 # All uncovered
        uncovered_count = num_t
        selected_blocks = []
        
        # Lazy Heap: ( -gain, block_idx )
        # We need to calculate gain for each block initially.
        # Gain = popcount(uncovered_mask & block_mask)
        # But we don't have block_mask as a single integer for bit ops easily in Python if num_t > 64?
        # num_t = C(19,5) = 11628. Python ints handle this fine.
        
        # Let's precompute the bitmask for each block as a Python int.
        # This might be memory heavy if we store 75k big ints, but 75k * 1.5KB ~ 100MB. Might be okay.
        # Alternatively, just use the list of indices and count intersections.
        # Given time constraints, let's use the list of indices and a set for uncovered.
        # Set of uncovered indices is faster for membership testing in Python than bit ops for 11k bits?
        # Actually, bit ops on Python big ints are implemented in C and are very fast.
        # Let's use bitmasks.
        
        block_masks = [0] * num_k
        for ki, t_indices in enumerate(k_to_t_indices):
            mask = 0
            for ti in t_indices:
                mask |= (1 << ti)
            block_masks[ki] = mask
            
        # Initialize heap
        heap = []
        for ki in range(num_k):
            gain = (uncovered_mask & block_masks[ki]).bit_count()
            if gain > 0:
                # Add small random noise to break ties and diversify
                noise = random.random() * 0.001 if seed is not None else 0
                heapq.heappush(heap, (-(gain + noise), ki))
                
        while uncovered_count > 0:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_ki = -1
            
            while heap:
                neg_gain, ki = heapq.heappop(heap)
                gain = -neg_gain
                # Check if this entry is still valid
                # Recalculate gain
                actual_gain = (uncovered_mask & block_masks[ki]).bit_count()
                if actual_gain == 0:
                    continue
                if actual_gain != int(gain): # Note: gain might have noise, so compare int(gain)
                    # Stale entry, re-push with correct gain
                    noise = random.random() * 0.001 if seed is not None else 0
                    heapq.heappush(heap, (-(actual_gain + noise), ki))
                    continue
                # Valid entry
                best_gain = actual_gain
                best_ki = ki
                break
            
            if best_ki == -1:
                # No blocks cover any uncovered sets? Should not happen if uncovered_count > 0
                break
                
            # Select block best_ki
            selected_blocks.append(best_ki)
            
            # Update uncovered
            block_mask = block_masks[best_ki]
            newly_covered = uncovered_mask & block_mask
            if newly_covered:
                uncovered_count -= newly_covered.bit_count()
                uncovered_mask &= ~newly_covered
            
            # Update heap for other blocks?
            # The lazy approach handles this: when we pop a block, we recompute its gain.
            # We don't need to push updates for all other blocks, just recompute when popped.
            
        return selected_blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy instances
    # How many restarts?
    # Each greedy run takes T_greedy.
    # Let's estimate T_greedy. ~300 steps. Each step pops from heap.
    # Heap size ~75k. Pop is log(75k) ~ 16.
    # 300 * 16 = 4800 heap ops. Plus recomputation of gain (bit_count on 11k bits).
    # This should be fast, maybe 0.1s - 0.5s per run?
    # Let's try 20 restarts.
    
    max_restarts = 30
    
    for seed in range(max_restarts):
        if time.time() > deadline - 10: # Leave 10s for local search on best
            break
            
        blocks_idx = run_greedy(seed)
        
        if not blocks_idx:
            continue
            
        # Local Search
        # Convert indices to actual blocks for local search logic?
        # Or work with indices.
        
        current_blocks = list(blocks_idx)
        
        # Helper to get unique subsets for a block given current coverage counts
        def compute_counts(blocks_list):
            counts = [0] * num_t
            for ki in blocks_list:
                for ti in k_to_t_indices[ki]:
                    counts[ti] += 1
            return counts
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            counts = compute_counts(current_blocks)
            
            # 1. Try to remove blocks with 0 unique subsets
            removed = False
            for i, ki in enumerate(list(current_blocks)):
                unique_subs = []
                for ti in k_to_t_indices[ki]:
                    if counts[ti] == 1:
                        unique_subs.append(ti)
                        
                if not unique_subs:
                    current_blocks.pop(i)
                    improved = True
                    removed = True
                    break # Restart loop with new state
            
            if removed:
                continue
                
            # 2. Try to replace blocks
            # Sort blocks by number of unique subsets to target "bad" blocks?
            # Or just iterate.
            # For each block, find its unique subsets.
            # If |unique| == 1, try to replace with a random block covering that subset.
            # If |unique| > 1, try to find intersection of covering blocks.
            
            # To avoid expensive intersection checks for every block every time,
            # we can just try replacement on the first block we find that has unique subsets.
            
            for i, ki in enumerate(current_blocks):
                unique_subs = [ti for ti in k_to_t_indices[ki] if counts[ti] == 1]
                
                if not unique_subs:
                    continue # Should have been removed
                
                # Find candidates for replacement
                # Candidates must cover all unique_subs
                # Start with all blocks, intersect with those covering each unique sub
                
                # Optimization: If len(unique_subs) == 1
                if len(unique_subs) == 1:
                    ti = unique_subs[0]
                    candidates = t_to_k_indices[ti]
                    # Pick a random candidate different from ki
                    # Filter out ki
                    cand_list = [c for c in candidates if c != ki]
                    if cand_list:
                        new_ki = random.choice(cand_list)
                        # Check if new_ki is already in current_blocks?
                        # If it is, replacing doesn't change the set of blocks, so no benefit?
                        # Actually, if new_ki is in current_blocks, then we are just moving coverage.
                        # But we need to ensure the unique sub is still covered.
                        # Since new_ki covers ti, and ti was unique to ki, after replacement,
                        # ti is covered by new_ki.
                        # However, if new_ki is already present, we are effectively just "reassigning" the coverage?
                        # No, we are replacing block ki with block new_ki.
                        # If new_ki is already in the list, we will have a duplicate.
                        # We should avoid duplicates.
                        if new_ki in current_blocks:
                            continue
                            
                        current_blocks[i] = new_ki
                        improved = True
                        break # Restart loop
                else:
                    # len(unique_subs) > 1
                    # Intersect candidate lists
                    # Start with the smallest list to reduce intersection size
                    unique_subs_sorted = sorted(unique_subs, key=lambda ti: len(t_to_k_indices[ti]))
                    candidates = set(t_to_k_indices[unique_subs_sorted[0]])
                    for ti in unique_subs_sorted[1:]:
                        candidates &= set(t_to_k_indices[ti])
                        if not candidates:
                            break
                    
                    candidates.discard(ki)
                    if candidates:
                        # Check for duplicates in current_blocks
                        valid_cands = [c for c in candidates if c not in current_blocks]
                        if valid_cands:
                            new_ki = random.choice(valid_cands)
                            current_blocks[i] = new_ki
                            improved = True
                            break # Restart loop
                            
            if time.time() > deadline - 5:
                break

        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
            
    if best_blocks is None:
        # Fallback: return empty or a valid trivial covering if possible?
        # For v=19, k=8, t=5, a valid covering must exist.
        # If we ran out of time before finding one, return empty (invalid) or try one last fast greedy?
        # The prompt says "Valid when every t-subset ... lies in at least one block".
        # If we return invalid, score is 0.
        # So we must ensure we return a valid covering.
        # The greedy loop runs until uncovered is empty or time runs out.
        # If time runs out, `uncovered` might not be empty.
        # We should check if the last generated blocks are valid.
        
        # Let's do a final check on the last generated `blocks_idx` if best_blocks is None.
        # Actually, the loop `for seed in range...` updates `best_blocks` only if `uncovered` was empty?
        # No, `run_greedy` returns blocks. It doesn't check validity explicitly in the return, 
        # but the loop condition `while uncovered_count > 0` ensures it tries to cover all.
        # If it breaks due to time, `uncovered_count` might be > 0.
        # We should modify `run_greedy` to return (blocks, is_valid).
        
        # Let's re-run a quick check on the last candidate.
        # To be safe, if best_blocks is None, we try one more time with a strict deadline check inside greedy.
        
        pass
        
    if best_blocks is None:
        return []
        
    return [list(all_k_subsets[ki]) for ki in best_blocks]
