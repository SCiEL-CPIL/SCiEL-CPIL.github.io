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
    block_covers = [[] for _ in range(num_k_subsets)]
    covers = [[] for _ in range(num_t_subsets)]
    
    for i, block in enumerate(k_subsets):
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            block_covers[i].append(idx)
            covers[idx].append(i)
            
    # Precompute masks for fast gain calculation in greedy
    block_masks = [0] * num_k_subsets
    for i in range(num_k_subsets):
        mask = 0
        for idx in block_covers[i]:
            mask |= (1 << idx)
        block_masks[i] = mask

    def greedy_cover(randomize=True):
        uncovered = (1 << num_t_subsets) - 1
        selected_blocks = []
        
        while uncovered:
            if time.time() > deadline - 20:
                break
                
            max_gain = 0
            best_candidates = []
            
            for i in range(num_k_subsets):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [i]
                elif gain == max_gain and max_gain > 0:
                    best_candidates.append(i)
                    
            if not best_candidates:
                break # Should not happen
                
            if randomize and len(best_candidates) > 1:
                # Pick randomly from top candidates to diversify
                best_block = random.choice(best_candidates)
            else:
                best_block = best_candidates[0]
                
            selected_blocks.append(best_block)
            uncovered &= ~block_masks[best_block]
            
        return selected_blocks

    best_solution = None
    best_len = float('inf')
    
    # Run multiple greedy passes with randomization
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        sol = greedy_cover(randomize=True)
        
        # Convert to set for fast membership
        sol_set = set(sol)
        
        # Local Search
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
                    for idx in block_covers[b]:
                        counts[idx] -= 1
                    current_sol.pop(i)
                    current_set.remove(b)
                    improved = True
                else:
                    i += 1
                    
            if len(current_sol) < best_len:
                best_len = len(current_sol)
                best_solution = current_sol[:]
                
            if not improved:
                # Phase 2: Try 1-Swap
                swap_found = False
                # Try a few random swaps
                for _ in range(50):
                    if not current_sol:
                        break
                    b_in = random.choice(current_sol)
                    unique_subs = []
                    for idx in block_covers[b_in]:
                        if counts[idx] == 1:
                            unique_subs.append(idx)
                            
                    if not unique_subs:
                        continue
                        
                    candidates = set(covers[unique_subs[0]])
                    for u in unique_subs[1:]:
                        candidates &= set(covers[u])
                        
                    candidates -= current_set
                    candidates.discard(b_in)
                    
                    if candidates:
                        b_out = random.choice(list(candidates))
                        # Perform swap
                        for idx in block_covers[b_in]:
                            counts[idx] -= 1
                        for idx in block_covers[b_out]:
                            counts[idx] += 1
                            
                        idx_in = current_sol.index(b_in)
                        current_sol[idx_in] = b_out
                        current_set.discard(b_in)
                        current_set.add(b_out)
                        
                        improved = True
                        swap_found = True
                        break
                        
                if swap_found:
                    continue
                    
                # Phase 3: Try 2-for-1 Swap (remove 2, add 1)
                # This is more expensive, so limit attempts
                two_swap_found = False
                if len(current_sol) >= 2:
                    # Sample a few pairs
                    for _ in range(20):
                        if time.time() > deadline - 5:
                            break
                        pair = random.sample(current_sol, 2)
                        b1, b2 = pair
                        
                        # Find unique subsets covered by b1 or b2
                        unique_subs = []
                        for idx in block_covers[b1]:
                            if counts[idx] == 1:
                                unique_subs.append(idx)
                        for idx in block_covers[b2]:
                            if counts[idx] == 1:
                                # Check if it's not already in unique_subs (it can't be, as counts are global)
                                unique_subs.append(idx)
                        
                        if not unique_subs:
                            # Both blocks are redundant? Phase 1 should have caught this.
                            # But if counts were updated, maybe.
                            # If unique_subs is empty, we can just remove both.
                            # This is a 2-for-0 swap.
                            for idx in block_covers[b1]:
                                counts[idx] -= 1
                            for idx in block_covers[b2]:
                                counts[idx] -= 1
                            
                            current_sol.remove(b1)
                            current_sol.remove(b2)
                            current_set.discard(b1)
                            current_set.discard(b2)
                            
                            improved = True
                            two_swap_found = True
                            break
                            
                        # Find b_out that covers all unique_subs
                        candidates = set(covers[unique_subs[0]])
                        for u in unique_subs[1:]:
                            candidates &= set(covers[u])
                            
                        candidates -= current_set
                        candidates.discard(b1)
                        candidates.discard(b2)
                        
                        if candidates:
                            b_out = random.choice(list(candidates))
                            # Perform 2-for-1 swap
                            for idx in block_covers[b1]:
                                counts[idx] -= 1
                            for idx in block_covers[b2]:
                                counts[idx] -= 1
                            for idx in block_covers[b_out]:
                                counts[idx] += 1
                                
                            current_sol.remove(b1)
                            current_sol.remove(b2)
                            current_sol.append(b_out)
                            current_set.discard(b1)
                            current_set.discard(b2)
                            current_set.add(b_out)
                            
                            improved = True
                            two_swap_found = True
                            break
                            
    if best_solution is None:
        # Fallback
        sol = greedy_cover(randomize=False)
        best_solution = sol
        best_len = len(sol)
        
    return [list(k_subsets[i]) for i in best_solution]
