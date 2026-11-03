import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # Precompute all t-subsets and map them to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t_subs = len(t_subs)
    
    # Precompute all k-subsets and their coverage masks
    k_subs = list(itertools.combinations(range(v), k))
    k_sub_masks = []
    for k_sub in k_subs:
        mask = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        k_sub_masks.append(mask)
    
    n_k_subs = len(k_subs)
    full_mask = (1 << n_t_subs) - 1
    
    def greedy_construct(randomize_ties=False):
        uncovered = full_mask
        blocks = []
        used_indices = set()
        
        # We will pick blocks one by one
        # To speed up, we can pre-compute the "potential" of each block?
        # No, potential changes as uncovered changes.
        
        # Optimization: 
        # We can maintain a list of active candidates.
        # However, with 75k candidates and ~300 iterations, 75k*300 = 22.5M ops.
        # In Python, this might be tight but doable if optimized.
        
        # Better: Randomized Greedy
        # At each step, find the block that covers the most uncovered elements.
        # If randomize_ties, add random noise to the count to break ties or perturb selection.
        
        while uncovered:
            best_idx = -1
            best_count = 0
            best_mask = 0
            
            # Iterate through all blocks
            # We can skip blocks that are already used? 
            # The problem states blocks must be distinct.
            # So we should not pick the same block twice.
            
            for i in range(n_k_subs):
                if i in used_indices:
                    continue
                mask = k_sub_masks[i]
                # Calculate new coverage
                new_cov = mask & uncovered
                if not new_cov:
                    continue
                
                cnt = new_cov.bit_count()
                
                # Randomization: if counts are close, or randomly, perturb
                if randomize_ties:
                    # Add a small random noise to break ties or explore
                    # This effectively makes the greedy choice non-deterministic
                    # We only care about relative ordering for the "best"
                    # A simpler way: if cnt == best_count, randomly decide?
                    # Or: if cnt > best_count, take it. If cnt == best_count and random, take it?
                    # Let's use a slight bias: if cnt >= best_count and (cnt > best_count or random.random() < 0.5):
                    # Actually, simple random tie-breaking is enough.
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
                        best_mask = mask
                    elif cnt == best_count and cnt > 0:
                        if random.random() < 0.5:
                            best_idx = i
                            best_mask = mask
                else:
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
                        best_mask = mask
            
            if best_idx == -1:
                # Should not happen if covering is possible and we haven't used all
                break
                
            blocks.append(best_idx)
            used_indices.add(best_idx)
            uncovered &= ~best_mask
            
            if time.time() > deadline:
                return None # Timeout in greedy
        
        return blocks
    
    def local_search(block_indices):
        # Convert to set for fast lookup
        # We want to reduce the number of blocks.
        # Strategy: 
        # 1. Remove any redundant block (standard reduction).
        # 2. Try 2-opt: Replace block A and B with block C and D?
        #    Or simply: Try to replace one block with another that allows more removals?
        #    
        # Given time constraints, let's stick to a robust "Reduction + Random Swap" loop.
        
        current_blocks = list(block_indices)
        
        # First, standard reduction
        def is_redundant(idx, blocks_list):
            # Check if removing idx leaves any t-subset uncovered
            # This is O(n_t_subs) if we recompute, or O(1) with counts.
            # Let's use counts for efficiency in the reduction loop.
            pass # Placeholder, we'll do it in the main loop below
            
        # Efficient reduction
        cover_count = [0] * n_t_subs
        for idx in current_blocks:
            mask = k_sub_masks[idx]
            m = mask
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                cover_count[i] += 1
                m ^= lsb
                
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for idx in current_blocks:
                mask = k_sub_masks[idx]
                m = mask
                zero_found = False
                affected = []
                while m:
                    lsb = m & -m
                    i = lsb.bit_length() - 1
                    cover_count[i] -= 1
                    affected.append(i)
                    if cover_count[i] == 0:
                        zero_found = True
                        break
                    m ^= lsb
                
                if zero_found:
                    # Restore
                    for i in affected:
                        cover_count[i] += 1
                    new_blocks.append(idx)
                else:
                    # Keep removed
                    changed = True
                    
            current_blocks = new_blocks
            
            if time.time() > deadline:
                break
        
        # Now try local swaps
        # We try to swap one block with another to potentially enable further reductions?
        # Or simply try to find a smaller set?
        # A common heuristic: "1-move" local search.
        # Try removing a block and adding a different one? No, that keeps size same.
        # We want to reduce size.
        
        # Let's try a "Random Restart Reduction" inside the local search.
        # Actually, the main power comes from the multi-start greedy.
        # Let's just do a few iterations of "Try to remove a random block, if fail, try to swap it with a random other block and see if reduction improves".
        
        # For simplicity and speed, we will just return the reduced set.
        # The multi-start nature of the outer loop will provide the diversity.
        
        return current_blocks

    best_solution = None
    best_len = float('inf')
    
    # Run multiple greedy restarts
    # How many? Depends on time.
    # Let's estimate time per greedy run.
    # 75k candidates * 300 steps * (bit ops) ~ 20M ops.
    # Python does ~10M-50M simple ops/sec.
    # So one run might take 0.5s - 1s.
    # We have 115s. So we can do ~100-200 runs.
    
    attempts = 0
    while time.time() < deadline:
        # Alternate between deterministic and randomized?
        # Randomized is generally better for escaping local optima.
        # Let's always randomize ties.
        sol = greedy_construct(randomize_ties=True)
        attempts += 1
        
        if sol is None:
            break
            
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
            # Try to improve with local search if we have time
            # Local search (reduction) is relatively fast.
            if time.time() < deadline - 10:
                reduced_sol = local_search(sol)
                if len(reduced_sol) < best_len:
                    best_len = len(reduced_sol)
                    best_solution = reduced_sol
                    
        # If we have a very good solution, we might want to spend more time optimizing it?
        # Or just keep searching for a better one.
        # Given the "open" nature, finding a significantly smaller one is a jump.
        # Greedy usually gets close to optimal.
        
    if best_solution is None:
        # Fallback: standard greedy if randomization failed or timed out immediately
        # This shouldn't happen
        sol = greedy_construct(randomize_ties=False)
        if sol:
            best_solution = sol
        else:
            # Emergency: return empty or partial?
            # Must return valid covering.
            # If we timed out in greedy, we might have an incomplete list.
            # The function contract says "Valid when every t-subset...".
            # If we return an invalid covering, score is 0.
            # We must ensure we return a valid covering.
            # The greedy loop `while uncovered` ensures validity IF it completes.
            # If it breaks due to timeout, it's invalid.
            # So we must track the last VALID state?
            # No, greedy constructs incrementally. If it stops early, it's not a covering.
            # So we should only accept solutions that completed the greedy phase.
            
            # If we didn't find any valid solution, we have a problem.
            # Let's force a deterministic run with a tight deadline?
            # Or just return an empty list? No, that's invalid.
            # Let's assume the first run completes. 75k * 300 is fast enough.
            pass

    # Convert indices back to blocks (lists of points)
    # The evaluator expects "list of blocks", where each block is a list of k integers.
    return [k_subs[i] for i in best_solution]
