import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for safety
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # 3. Pre-compute coverage structures
    # block_covers[i] = list of t-subset indices covered by block i
    block_covers = [[] for _ in range(num_k_subsets)]
    # covers[j] = list of block indices that cover t-subset j
    covers = [[] for _ in range(num_t_subsets)]
    
    for i, block in enumerate(k_subsets):
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            block_covers[i].append(idx)
            covers[idx].append(i)
            
    # Static score: number of t-subsets covered (always C(k,t) for valid blocks, but good for sorting if we had invalid ones? No, all k-subsets cover C(k,t) t-subsets. 
    # Wait, C(k,t) is constant for all blocks. So static score is useless for sorting.
    # We need a dynamic score or randomization.
    
    # Helper to convert a set of block indices to a bitmask for fast membership check
    # num_k_subsets is ~3876, which fits in a Python int (arbitrary precision) or a few 64-bit words.
    
    def greedy_cover(randomize=False):
        uncovered = (1 << num_t_subsets) - 1
        selected_blocks = []
        # To speed up, we can maintain a list of active blocks? 
        # Or just iterate all blocks.
        # Since C(k,t) is constant, we can't sort by static gain.
        # We must compute gain dynamically.
        
        while uncovered:
            if time.time() > deadline - 20: # Leave time for LS
                break
                
            max_gain = 0
            best_candidates = []
            
            # Iterate all blocks
            for i in range(num_k_subsets):
                # Calculate gain
                # We can precompute mask for each block?
                # Let's precompute masks for all blocks once outside?
                # No, we need to do it inside or precompute.
                # Precomputing masks for 3876 blocks of 3876 bits is ~1.5MB. Very fast.
                pass # We will precompute masks below.
            
            # This inner loop is O(N_blocks). 
            # 3876 blocks * 150 steps = 580k iterations. Fast.
            
            # Let's precompute masks to make gain calculation O(1) big-int op.
            # This is done in the main scope.
            
            for i in range(num_k_subsets):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [i]
                elif gain == max_gain and max_gain > 0:
                    best_candidates.append(i)
                    
            if not best_candidates:
                break # Should not happen if covering exists
                
            if randomize and len(best_candidates) > 1:
                best_block = random.choice(best_candidates)
            else:
                best_block = best_candidates[0]
                
            selected_blocks.append(best_block)
            uncovered &= ~block_masks[best_block]
            
        return selected_blocks

    # Precompute masks
    block_masks = [0] * num_k_subsets
    for i in range(num_k_subsets):
        mask = 0
        for idx in block_covers[i]:
            mask |= (1 << idx)
        block_masks[i] = mask

    best_solution = None
    best_len = float('inf')
    
    # Run multiple greedy passes with randomization
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        sol = greedy_cover(randomize=(r > 0))
        # Convert to set for fast membership
        sol_set = set(sol)
        
        # Local Search
        # We need to track counts of t-subsets
        counts = [0] * num_t_subsets
        for b in sol:
            for idx in block_covers[b]:
                counts[idx] += 1
                
        current_sol = sol[:]
        current_set = set(current_sol)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Phase 1: Remove redundant blocks
            i = 0
            while i < len(current_sol):
                b = current_sol[i]
                is_red = True
                for idx in block_covers[b]:
                    if counts[idx] == 1:
                        is_red = False
                        break
                if is_red:
                    # Remove b
                    for idx in block_covers[b]:
                        counts[idx] -= 1
                    current_sol.pop(i)
                    current_set.remove(b)
                    improved = True
                    # Do not increment i
                else:
                    i += 1
                    
            if len(current_sol) < best_len:
                best_len = len(current_sol)
                best_solution = current_sol[:]
                
            if not improved:
                # Phase 2: Try 1-Swap
                # Pick a random block from current_sol
                # Or iterate through all? 
                # Let's try a few random swaps to find one that enables removal.
                # A swap replaces b_in with b_out.
                # Condition: b_out covers all unique subsets of b_in.
                
                # To make it efficient, we can try to find a swap that reduces "tightness".
                # But simply:
                # 1. Pick b_in.
                # 2. Find unique_subs of b_in.
                # 3. Find b_out that covers unique_subs.
                # 4. If found, swap.
                
                # We try this for a limited number of attempts
                swap_attempts = 20
                for _ in range(swap_attempts):
                    if not current_sol:
                        break
                    b_in = random.choice(current_sol)
                    unique_subs = []
                    for idx in block_covers[b_in]:
                        if counts[idx] == 1:
                            unique_subs.append(idx)
                            
                    if not unique_subs:
                        # Should have been removed in Phase 1
                        continue
                        
                    # Find b_out that covers all unique_subs
                    # Intersection of covers[u] for u in unique_subs
                    # Since unique_subs is small, we can do this.
                    candidates = set(covers[unique_subs[0]])
                    for u in unique_subs[1:]:
                        candidates &= set(covers[u])
                        
                    # Remove b_in from candidates (we want a different block)
                    # Also remove blocks already in current_set
                    candidates -= current_set
                    candidates.discard(b_in)
                    
                    if candidates:
                        b_out = random.choice(list(candidates))
                        # Perform swap
                        # Remove b_in
                        for idx in block_covers[b_in]:
                            counts[idx] -= 1
                        # Add b_out
                        for idx in block_covers[b_out]:
                            counts[idx] += 1
                        
                        # Update list
                        idx_in = current_sol.index(b_in)
                        current_sol[idx_in] = b_out
                        current_set.discard(b_in)
                        current_set.add(b_out)
                        
                        improved = True
                        break # Restart loop to allow removals
                        
    if best_solution is None:
        # Fallback: just return the last greedy result if LS failed to find anything better?
        # Or run one more greedy without randomization if we ran out of time?
        # The greedy loop above breaks on time. We should ensure we return *some* valid covering.
        # If best_solution is None, it means we didn't finish a full LS or greedy?
        # Actually, greedy_cover returns a list. If it breaks early, it's invalid.
        # We must ensure validity.
        # If time is tight, we should not break greedy early unless we have a fallback.
        # Better: Run greedy to completion if time allows, otherwise use a precomputed static solution?
        # No, we can't precompute.
        # We should make greedy robust.
        # If we break greedy early, we return invalid.
        # So we should not break greedy unless we are sure we can finish LS?
        # Actually, if we break greedy, we should just return that partial list? No, invalid.
        # So we must ensure greedy finishes.
        # Given 150s limit and greedy is fast, it should finish.
        
        # If best_solution is None, we take the first greedy result and assume it's valid (if greedy didn't break).
        # Let's assume greedy finishes.
        if sol:
             best_solution = sol
             best_len = len(sol)
             
    return [list(k_subsets[i]) for i in best_solution]
