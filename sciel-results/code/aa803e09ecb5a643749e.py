import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    # The evaluator kills after 150s. We should aim for ~140s to be safe.
    deadline = start_time + 135.0
    
    # 1. Precomputation
    # Map t-subsets to IDs
    t_subs = list(itertools.combinations(range(v), t))
    t_to_id = {ts: i for i, ts in enumerate(t_subs)}
    n_t = len(t_subs)
    full_mask = (1 << n_t) - 1
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_blocks)
    
    # Precompute masks for each block
    # block_masks[i] is a bitset of t-subset IDs covered by block i
    block_masks = [0] * n_k
    
    # Precompute which blocks cover each t-subset
    # t_to_blocks[j] is a list of block indices that cover t-subset j
    t_to_blocks = [[] for _ in range(n_t)]
    
    # Optimization: Precompute the list of t-subset IDs for each block
    # block_t_ids[i] is a list of t-subset IDs covered by block i
    block_t_ids = [[] for _ in range(n_k)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        ids = []
        for t_sub in itertools.combinations(block, t):
            tid = t_to_id[t_sub]
            mask |= (1 << tid)
            t_to_blocks[tid].append(i)
            ids.append(tid)
        block_masks[i] = mask
        block_t_ids[i] = ids
        
    if time.time() > deadline - 10:
        # Fallback if precomputation is too slow (unlikely for these sizes)
        return greedy_cover_fallback(v, k, t, block_masks, t_to_id, full_mask, all_blocks)

    # Helper: Greedy initialization
    def greedy_init(randomize=False):
        covered = 0
        solution = []
        
        while covered != full_mask:
            best_block = -1
            best_gain = 0
            
            if randomize:
                # Collect top candidates
                candidates = []
                for i in range(n_k):
                    gain = (block_masks[i] & ~covered).bit_count()
                    if gain > 0:
                        candidates.append((gain, i))
                if not candidates:
                    break
                # Sort by gain descending, take top 10, pick random
                candidates.sort(key=lambda x: x[0], reverse=True)
                top_k = candidates[:10]
                # Weighted random? Or just random from top?
                chosen = random.choice(top_k)[1]
                solution.append(chosen)
                covered |= block_masks[chosen]
            else:
                for i in range(n_k):
                    gain = (block_masks[i] & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = i
                if best_block == -1:
                    break
                solution.append(best_block)
                covered |= block_masks[best_block]
                
            if time.time() > deadline - 5:
                break
        return solution

    # Helper: Local Search with Counts
    def local_search_with_counts(solution):
        sol_set = set(solution)
        n_t = len(t_subs)
        counts = [0] * n_t
        
        # Initialize counts
        for b in solution:
            for tid in block_t_ids[b]:
                counts[tid] += 1
        
        improved = True
        while improved:
            improved = False
            # Iterate over blocks in solution
            for i in range(len(solution)):
                if time.time() > deadline - 5:
                    return solution
                
                b = solution[i]
                
                # Check if b is redundant
                is_redundant = True
                for tid in block_t_ids[b]:
                    if counts[tid] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove b
                    for tid in block_t_ids[b]:
                        counts[tid] -= 1
                    solution.pop(i)
                    sol_set.remove(b)
                    improved = True
                    break # Restart pass
                
                # If not redundant, try 1-for-1 swap
                unique_tids = []
                for tid in block_t_ids[b]:
                    if counts[tid] == 1:
                        unique_tids.append(tid)
                
                if not unique_tids:
                    continue
                
                # Find a block c (c != b) that covers all unique_tids
                unique_tids.sort(key=lambda x: len(t_to_blocks[x]))
                
                # Start with the most constrained tid
                candidates = set(t_to_blocks[unique_tids[0]])
                candidates.discard(b)
                
                for tid in unique_tids[1:]:
                    candidates.intersection_update(t_to_blocks[tid])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Choose a candidate
                c_in_sol = [c for c in candidates if c in sol_set]
                if c_in_sol:
                    # Just remove b
                    for tid in block_t_ids[b]:
                        counts[tid] -= 1
                    solution.pop(i)
                    sol_set.remove(b)
                    improved = True
                    break # Restart
                
                c_new = [c for c in candidates if c not in sol_set]
                if c_new:
                    # Pick one randomly
                    c = random.choice(c_new)
                    
                    # Perform swap
                    for tid in block_t_ids[b]:
                        counts[tid] -= 1
                    for tid in block_t_ids[c]:
                        counts[tid] += 1
                    
                    solution[i] = c
                    sol_set.remove(b)
                    sol_set.add(c)
                    improved = True
                    break # Restart pass
                    
        return solution

    # Run Initial Greedy
    best_solution = greedy_init(randomize=False)
    best_count = len(best_solution)
    
    # Run Local Search
    current_solution = local_search_with_counts(best_solution)
    if len(current_solution) < best_count:
        best_solution = current_solution
        best_count = len(best_solution)
    
    # Iterated Local Search
    while time.time() < deadline - 10:
        # Randomized Greedy
        sol = greedy_init(randomize=True)
        
        # Local Search
        sol = local_search_with_counts(sol)
        
        if len(sol) < best_count:
            best_solution = sol
            best_count = len(best_solution)
            
    return [list(all_blocks[b]) for b in best_solution]

def greedy_cover_fallback(v, k, t, block_masks, t_to_id, full_mask, all_blocks):
    covered = 0
    solution = []
    while covered != full_mask:
        best_block = -1
        best_gain = 0
        for i, mask in enumerate(block_masks):
            gain = (mask & ~covered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = i
        if best_block == -1:
            break
        solution.append(best_block)
        covered |= block_masks[best_block]
    return solution
