import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe under 120s
    
    # 1. Map triples to indices
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # 2. Generate all candidate blocks and their masks
    # For v=24, k=7, num_blocks = 346104.
    blocks = []
    block_masks = []
    
    for block in itertools.combinations(range(v), k):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
        blocks.append(block)
        block_masks.append(mask)
        
    num_blocks = len(blocks)
    
    # Precompute block lists for each triple for fast intersection lookups
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i in range(num_blocks):
        mask = block_masks[i]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            triple_to_blocks[idx].append(i)
            m ^= lsb
            
    # Convert masks to a list of lists of triple indices for faster NumPy operations?
    # Or keep as integers for bit counting?
    # NumPy doesn't have efficient bit_count for large ints in older versions without ufuncs.
    # However, we can represent the "uncovered" state as a boolean array and compute gains via dot product?
    # Let's try a different vectorization approach:
    # Represent each block as a vector of 0s and 1s of length num_triples?
    # 346k blocks * 2024 triples = 700M entries. Too large for memory (5GB+).
    
    # Alternative: Keep integer masks but use Python's bit_count in a list comprehension?
    # That's still Python loop.
    
    # Let's stick to the Python loop for gain calculation but optimize the inner logic?
    # Actually, the current code loops 346k times in Python per greedy step.
    # If we do 100 greedy steps, that's 34M iterations. Might be okay in 110s?
    # 34M simple bit operations in Python might take ~5-10s.
    # 5 restarts * 100 steps * 346k = 173M iterations. This might be too slow.
    
    # Let's try to speed up the greedy step by pre-computing something?
    # Or use a "lazy" update?
    
    # Let's try to optimize the existing greedy by using `int.bit_count` which is fast in C.
    # The bottleneck is the Python loop overhead.
    
    # Strategy:
    # 1. Fast Greedy with randomized tie-breaking.
    # 2. Aggressive Local Search: 
    #    - Remove redundant blocks.
    #    - Try to replace 2 blocks with 1 block.
    
    def run_greedy(seed_val):
        random.seed(seed_val)
        uncovered = (1 << num_triples) - 1
        selected_indices = []
        
        while uncovered:
            if time.time() > deadline - 10:
                return None # Timeout buffer
                
            best_gain = -1
            candidates = []
            
            # Optimization: If the number of uncovered triples is small, 
            # we can filter candidates earlier?
            
            for i in range(num_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
            
            if best_gain == 0:
                return selected_indices
                
            # Randomly pick from best candidates
            best_idx = random.choice(candidates)
            selected_indices.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected_indices

    def refine_indices(selected_indices):
        if selected_indices is None:
            return []
            
        solution_set = set(selected_indices)
        
        coverage_count = [0] * num_triples
        for idx in solution_set:
            mask = block_masks[idx]
            m = mask
            while m:
                lsb = m & -m
                idx_t = lsb.bit_length() - 1
                coverage_count[idx_t] += 1
                m ^= lsb
                
        changed = True
        while changed:
            if time.time() > deadline - 5:
                break
            changed = False
            for idx in list(solution_set):
                mask = block_masks[idx]
                is_redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    if coverage_count[idx_t] <= 1:
                        is_redundant = False
                        break
                    m ^= lsb
                
                if is_redundant:
                    solution_set.remove(idx)
                    m = mask
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        coverage_count[idx_t] -= 1
                        m ^= lsb
                    changed = True
                    
        return list(solution_set)

    def local_search_2exchange(solution_indices):
        """
        Attempts to reduce the number of blocks by replacing pairs of blocks
        with a single block that covers all 'unique' triples of the pair.
        Optimized for speed.
        """
        if solution_indices is None:
            return []
            
        sol_set = set(solution_indices)
        if not sol_set:
            return []
            
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            
            # Recalculate coverage counts
            coverage_count = [0] * num_triples
            for idx in sol_set:
                mask = block_masks[idx]
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    coverage_count[idx_t] += 1
                    m ^= lsb
            
            sol_list = list(sol_set)
            
            # Iterate through pairs
            # To speed up, we can shuffle the list to avoid bias
            random.shuffle(sol_list)
            
            for i in range(len(sol_list)):
                for j in range(i + 1, len(sol_list)):
                    if time.time() > deadline - 5:
                        return [blocks[k] for k in sol_set]
                        
                    b1 = sol_list[i]
                    b2 = sol_list[j]
                    
                    mask1 = block_masks[b1]
                    mask2 = block_masks[b2]
                    mask_pair = mask1 | mask2
                    
                    # Find critical triples:
                    # 1. Covered by b1 only (count 1, in mask1)
                    # 2. Covered by b2 only (count 1, in mask2)
                    # 3. Covered by b1 and b2 only (count 2, in mask1 & mask2)
                    
                    critical_triples = []
                    m = mask_pair
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        c = coverage_count[idx_t]
                        
                        if c == 1:
                            # Must be in exactly one of b1, b2
                            # If in mask1 and not mask2, and count 1 -> critical
                            # If in mask2 and not mask1, and count 1 -> critical
                            # Since it's in mask_pair, it's in at least one.
                            critical_triples.append(idx_t)
                        elif c == 2:
                            # Must be in both b1 and b2 to be critical for the pair removal
                            if (mask1 & lsb) and (mask2 & lsb):
                                critical_triples.append(idx_t)
                        m ^= lsb
                    
                    if not critical_triples:
                        # Both blocks are fully covered by others
                        sol_set.remove(b1)
                        sol_set.remove(b2)
                        improved = True
                        break
                    else:
                        # Try to find a block that covers all critical_triples
                        # Use the triple with the fewest covering blocks to filter
                        critical_triples.sort(key=lambda x: len(triple_to_blocks[x]))
                        
                        candidates = set(triple_to_blocks[critical_triples[0]])
                        
                        for t_idx in critical_triples[1:]:
                            candidates.intersection_update(triple_to_blocks[t_idx])
                            if not candidates:
                                break
                            
                        if candidates:
                            # Check if any candidate is NOT in sol_set
                            for b_k in candidates:
                                if b_k not in sol_set:
                                    sol_set.remove(b1)
                                    sol_set.remove(b2)
                                    sol_set.add(b_k)
                                    improved = True
                                    break
                            
                            if improved:
                                break
                            
            # If we broke out of the inner loop due to improvement, 
            # the 'improved' flag is True and the while loop restarts.
            # If we finished the loops without improvement, 'improved' remains False.
            
        return list(sol_set)

    best_solution_indices = None
    best_size = float('inf')
    
    # Run multiple restarts
    # We have ~110s. 
    # Greedy is the bottleneck. 
    # Let's allow 3 restarts to ensure we finish.
    num_restarts = 3
    
    for i in range(num_restarts):
        if time.time() > deadline - 30: # Leave 30s for local search on best
            break
            
        greedy_result = run_greedy(i)
        if greedy_result is None:
            continue
            
        refined_indices = refine_indices(greedy_result)
        
        if len(refined_indices) < best_size:
            best_size = len(refined_indices)
            best_solution_indices = refined_indices
            
    if best_solution_indices is None:
        # Fallback
        greedy_result = run_greedy(0)
        best_solution_indices = refine_indices(greedy_result) if greedy_result else []

    if best_solution_indices is not None and time.time() < deadline - 5:
        # Run local search on the best greedy result
        final_indices = local_search_2exchange(best_solution_indices)
        if final_indices and len(final_indices) < len(best_solution_indices):
            best_solution_indices = final_indices
            
    # Convert indices to blocks
    best_solution = [blocks[i] for i in best_solution_indices]
    best_solution.sort()
    return [list(b) for b in best_solution]
