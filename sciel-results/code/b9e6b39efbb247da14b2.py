import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(all_k_subsets)
    
    # Precompute for each block:
    # 1. The mask of t-subsets it covers
    # 2. The list of indices of t-subsets it covers (for fast count updates)
    block_mask = [0] * num_k
    block_t_indices = [[] for _ in range(num_k)]
    
    # covering_blocks[i] is a list of block indices that cover t-subset i
    covering_blocks = [[] for _ in range(num_t)]
    
    for b_idx, block in enumerate(all_k_subsets):
        mask = 0
        indices = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            indices.append(idx)
            covering_blocks[idx].append(b_idx)
        block_mask[b_idx] = mask
        block_t_indices[b_idx] = indices
        
    # Helper to get unique mask from counts
    # We will maintain unique_mask incrementally in the main loops to save time
    # But for initialization or full reset, we can compute it.
    
    def greedy_construct(seed=None):
        if seed is not None:
            random.seed(seed)
            
        active_set = set()
        counts = [0] * num_t
        uncovered_mask = (1 << num_t) - 1
        unique_mask = 0 # Not strictly needed for greedy, but good for consistency
        
        while uncovered_mask:
            best_block = -1
            max_gain = 0
            
            # Optimization: Shuffle the order of block evaluation to diversify
            # We don't want to shuffle all 50k blocks every step (too slow).
            # Instead, we can just pick randomly among ties or use a fixed order 
            # but randomize the initial seed.
            # Actually, standard greedy is deterministic. 
            # To diversify, we can add a small random noise to the gain or 
            # randomly break ties.
            
            for b_idx in range(num_k):
                gain = (block_mask[b_idx] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
                elif gain == max_gain and gain > 0:
                    # Random tie-breaking
                    if random.random() < 0.5:
                        best_block = b_idx
            
            if best_block == -1:
                break
                
            active_set.add(best_block)
            
            # Update counts and unique_mask
            for idx in block_t_indices[best_block]:
                old_count = counts[idx]
                counts[idx] = old_count + 1
                if old_count == 0:
                    # Newly covered, not unique yet (count becomes 1, so it IS unique)
                    unique_mask |= (1 << idx)
                elif old_count == 1:
                    # Was unique, now count 2, so no longer unique
                    unique_mask &= ~(1 << idx)
                # if old_count > 1, no change to unique status
            
            uncovered_mask &= ~block_mask[best_block]
            
        return active_set, counts, unique_mask

    def minimize(active_set, counts, unique_mask):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Iterate over a copy
            for b_idx in list(active_set):
                if b_idx not in active_set:
                    continue
                # Check if b_idx has any unique coverage
                if block_mask[b_idx] & unique_mask:
                    continue
                
                # It has no unique coverage, so it can be removed
                active_set.remove(b_idx)
                for idx in block_t_indices[b_idx]:
                    old_count = counts[idx]
                    counts[idx] = old_count - 1
                    if old_count == 2:
                        # Was 2, now 1, so becomes unique
                        unique_mask |= (1 << idx)
                    elif old_count == 1:
                        # Was 1, now 0, so removed from unique (and uncovered)
                        # But wait, if we remove a block with no unique coverage,
                        # it means all its subsets are covered by >= 2 blocks.
                        # So old_count must be >= 2.
                        # So unique_mask bit should be 0.
                        # If old_count was 1, it would have been unique, so we wouldn't be here.
                        pass
                
                changed = True
        return active_set, counts, unique_mask

    def local_search(active_set, counts, unique_mask):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            items = list(active_set)
            random.shuffle(items)
            
            for b_idx in items:
                if b_idx not in active_set:
                    continue
                
                # Check if removable
                u_b = block_mask[b_idx] & unique_mask
                if u_b == 0:
                    # Remove
                    active_set.remove(b_idx)
                    for idx in block_t_indices[b_idx]:
                        old_count = counts[idx]
                        counts[idx] = old_count - 1
                        if old_count == 2:
                            unique_mask |= (1 << idx)
                    improved = True
                    # After removal, minimize again?
                    # The removal might make other blocks removable.
                    # We can just continue the loop, the 'improved' flag restarts the pass.
                    # But we should probably run minimize to clean up immediately?
                    # Or just let the next pass handle it.
                    # Let's just set improved and break inner loop to restart pass?
                    # Or continue. If we continue, we use stale unique_mask?
                    # No, we updated unique_mask.
                    # But other blocks might now be removable.
                    # The while loop will catch them in the next iteration.
                    continue
                
                # Find swap candidates
                # We need to cover u_b with a new block.
                # Find the rarest subset in u_b
                bits = []
                temp = u_b
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    bits.append(idx)
                    temp ^= lsb
                
                if not bits:
                    continue
                
                # Sort by frequency (length of covering_blocks)
                bits.sort(key=lambda i: len(covering_blocks[i]))
                rarest_idx = bits[0]
                
                found_swap = False
                # Iterate candidates covering the rarest subset
                # Shuffle candidates to diversify
                cands = list(covering_blocks[rarest_idx])
                random.shuffle(cands)
                
                for cand_idx in cands:
                    if cand_idx == b_idx:
                        continue
                    if cand_idx in active_set:
                        continue
                    
                    # Check if cand covers all of u_b
                    if (block_mask[cand_idx] & u_b) == u_b:
                        # Perform swap
                        active_set.remove(b_idx)
                        active_set.add(cand_idx)
                        
                        # Update counts and unique_mask
                        # Remove b_idx
                        for idx in block_t_indices[b_idx]:
                            old_count = counts[idx]
                            counts[idx] = old_count - 1
                            if old_count == 2:
                                unique_mask |= (1 << idx)
                            elif old_count == 1:
                                unique_mask &= ~(1 << idx)
                        
                        # Add cand_idx
                        for idx in block_t_indices[cand_idx]:
                            old_count = counts[idx]
                            counts[idx] = old_count + 1
                            if old_count == 0:
                                unique_mask |= (1 << idx)
                            elif old_count == 1:
                                unique_mask &= ~(1 << idx)
                        
                        improved = True
                        found_swap = True
                        break # Break candidate loop, restart pass
                
                if found_swap:
                    break # Break item loop, restart pass

        return active_set, counts, unique_mask

    best_solution = None
    best_count = float('inf')
    
    # Run multiple restarts
    # Estimate time per restart.
    # Greedy: ~150 steps * 50k blocks * bit_count.
    # Local Search: Hard to estimate.
    # Let's try 10-20 restarts.
    
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        active_set, counts, unique_mask = greedy_construct(seed=i)
        active_set, counts, unique_mask = minimize(active_set, counts, unique_mask)
        active_set, counts, unique_mask = local_search(active_set, counts, unique_mask)
        
        current_count = len(active_set)
        if current_count < best_count:
            best_count = current_count
            best_solution = set(active_set)
            
    if best_solution is None:
        # Fallback
        active_set, counts, unique_mask = greedy_construct(seed=0)
        active_set, counts, unique_mask = minimize(active_set, counts, unique_mask)
        best_solution = active_set
        
    result = [list(all_k_subsets[b_idx]) for b_idx in best_solution]
    return result
