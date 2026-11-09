import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer

    # 1. Precomputation
    t_subs = list(itertools.combinations(range(v), t))
    t_to_id = {ts: i for i, ts in enumerate(t_subs)}
    n_t = len(t_subs)
    full_mask = (1 << n_t) - 1
    
    blocks = list(itertools.combinations(range(v), k))
    n_k = len(blocks)
    
    block_masks = [0] * n_k
    # For fast candidate finding: map tid to list of blocks
    t_to_blocks = [[] for _ in range(n_t)]
    
    for i, block in enumerate(blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            tid = t_to_id[t_sub]
            mask |= (1 << tid)
            t_to_blocks[tid].append(i)
        block_masks[i] = mask

    if time.time() > deadline - 10:
        # Emergency fallback
        return simple_greedy(v, k, t, block_masks, full_mask)

    def get_missing_mask(mask):
        return full_mask & ~mask

    def greedy_init():
        covered = 0
        solution = []
        # Use a set for fast membership if needed, but list is fine for small N
        while covered != full_mask:
            best_gain = 0
            candidates = []
            for i in range(n_k):
                gain = (block_masks[i] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            if not candidates:
                break
            # Random choice from best candidates to introduce diversity
            chosen = random.choice(candidates)
            solution.append(chosen)
            covered |= block_masks[chosen]
            if time.time() > deadline - 20:
                break
        return solution

    def local_search(solution):
        if not solution:
            return solution
        
        # Initialize coverage counts
        coverage_counts = [0] * n_t
        for b in solution:
            mask = block_masks[b]
            # Iterate through set bits
            m = mask
            while m:
                lsb = m & -m
                tid = lsb.bit_length() - 1
                coverage_counts[tid] += 1
                m ^= lsb
        
        sol_set = set(solution)
        
        max_iters = 50 # Safety valve for inner loop
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            changed = False
            # Shuffle solution to avoid bias
            shuffled_sol = solution.copy()
            random.shuffle(shuffled_sol)
            
            for b in shuffled_sol:
                if time.time() > deadline - 5:
                    break
                
                # Check if b is removable
                # Decrement counts
                mask_b = block_masks[b]
                missing_tids = []
                m = mask_b
                while m:
                    lsb = m & -m
                    tid = lsb.bit_length() - 1
                    coverage_counts[tid] -= 1
                    if coverage_counts[tid] == 0:
                        missing_tids.append(tid)
                    m ^= lsb
                
                if not missing_tids:
                    # Can remove b
                    solution.remove(b)
                    sol_set.remove(b)
                    changed = True
                    break # Restart pass
                else:
                    # Need to find a replacement b' not in sol_set covering all missing_tids
                    # Find intersection of t_to_blocks for missing_tids
                    # Start with the smallest list
                    missing_tids.sort(key=lambda x: len(t_to_blocks[x]))
                    if not t_to_blocks[missing_tids[0]]:
                        # No block covers this tid? Impossible if valid covering exists, 
                        # but might happen if we are in a broken state? 
                        # We start with a valid covering, so this shouldn't happen.
                        pass
                    
                    candidates = set(t_to_blocks[missing_tids[0]])
                    for tid in missing_tids[1:]:
                        candidates.intersection_update(t_to_blocks[tid])
                        if not candidates:
                            break
                    
                    # Filter out current solution blocks
                    valid_candidates = [c for c in candidates if c not in sol_set]
                    
                    if valid_candidates:
                        # Pick one
                        chosen = random.choice(valid_candidates)
                        # Replace b with chosen
                        solution.remove(b)
                        sol_set.remove(b)
                        solution.append(chosen)
                        sol_set.add(chosen)
                        
                        # Update coverage counts for chosen
                        m = block_masks[chosen]
                        while m:
                            lsb = m & -m
                            tid = lsb.bit_length() - 1
                            coverage_counts[tid] += 1
                            m ^= lsb
                        # Note: We already decremented for b. 
                        # The counts for missing_tids are now 1 (from chosen).
                        # Counts for other tids in chosen are incremented.
                        # This is correct.
                        
                        changed = True
                        break # Restart pass
                    else:
                        # No replacement found, restore counts for b
                        m = mask_b
                        while m:
                            lsb = m & -m
                            tid = lsb.bit_length() - 1
                            coverage_counts[tid] += 1
                            m ^= lsb
            
            if not changed:
                break
        
        return solution

    best_solution = None
    best_count = float('inf')
    
    # Number of restarts depends on time
    # Precomp took ~2s. 138s left.
    # Let's try 10 restarts.
    for i in range(10):
        if time.time() > deadline - 10:
            break
        
        sol = greedy_init()
        # Ensure valid? Greedy should produce valid.
        # Local search maintains validity.
        sol = local_search(sol)
        
        if len(sol) < best_count:
            best_count = len(sol)
            best_solution = sol.copy()
            
    if best_solution is None:
        # Fallback
        return simple_greedy(v, k, t, block_masks, full_mask)
        
    return [list(blocks[b]) for b in best_solution]

def simple_greedy(v, k, t, block_masks, full_mask):
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
