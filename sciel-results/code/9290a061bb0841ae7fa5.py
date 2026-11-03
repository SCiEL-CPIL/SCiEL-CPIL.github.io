import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0 # Safety margin

    def elapsed():
        return time.time() - start_time

    # 1. Precomputation
    # Generate all t-sets and assign indices
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Generate all blocks and their bitmasks
    all_blocks = []
    block_masks = []
    for block in itertools.combinations(range(v), k):
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << tset_index[ts])
        all_blocks.append(block)
        block_masks.append(mask)
        
    num_blocks = len(all_blocks)
    
    # Initial full mask
    full_mask = (1 << num_tsets) - 1

    # 2. Greedy Set Cover
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        used = [False] * num_blocks
        
        while uncovered:
            if elapsed() > deadline - 5:
                break
            best_idx = -1
            best_count = -1
            
            for i in range(num_blocks):
                if used[i]:
                    continue
                count = (block_masks[i] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
                    if count == num_tsets:
                        break
            
            if best_idx == -1 or best_count == 0:
                break
                
            used[best_idx] = True
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected

    # 3. Local Search (1-for-1 swap and deletion)
    def local_search(initial_sol, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        
        # Precompute cover counts for current solution
        # cover_counts[ts_idx] = number of blocks in sol covering ts_idx
        # Using a dict or array? Array of size 11628 is fine.
        cover_counts = [0] * num_tsets
        
        def update_counts(block_idx, delta):
            mask = block_masks[block_idx]
            # Iterate through set bits in mask
            m = mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_counts[idx] += delta
                m ^= lsb

        # Initialize counts
        for b in sol:
            update_counts(b, 1)
            
        best_sol = list(sol)
        best_size = len(sol)
        no_improve = 0
        max_no_improve = 50
        
        while elapsed() < deadline - 10 and no_improve < max_no_improve:
            improved = False
            
            # Phase 1: Try to remove redundant blocks
            # A block is redundant if all its t-sets are covered by others.
            # i.e., for all ts in block, cover_counts[ts] > 1.
            # We can check this by iterating bits in block_mask.
            
            # Shuffle order to avoid bias
            order = list(range(len(sol)))
            rng.shuffle(order)
            
            for pos in order:
                if pos >= len(sol):
                    continue
                b = sol[pos]
                mask = block_masks[b]
                redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                    
                if redundant:
                    # Remove block
                    update_counts(b, -1)
                    sol.pop(pos)
                    improved = True
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    # Restart loop to check for more redundancies
                    break
                    
            if improved:
                no_improve = 0
                continue
                
            # Phase 2: 1-for-1 Swap
            # Pick a block A to remove.
            # Let U be the set of t-sets in A that become uncovered (cover_counts == 1).
            # Find a block B (not in sol) that covers all t-sets in U.
            # If found, swap A with B.
            
            if sol:
                found_swap = False
                # Try a few random blocks to remove
                for _ in range(10):
                    if not sol:
                        break
                    pos = rng.randint(0, len(sol) - 1)
                    b_out = sol[pos]
                    
                    # Find critical t-sets for b_out
                    critical_mask = 0
                    m = block_masks[b_out]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] == 1:
                            critical_mask |= lsb
                        m ^= lsb
                        
                    if critical_mask == 0:
                        # Should have been caught in Phase 1, but just in case
                        update_counts(b_out, -1)
                        sol.pop(pos)
                        improved = True
                        found_swap = True
                        break
                        
                    # Find a block covering critical_mask
                    # We need a block B such that (block_masks[B] & critical_mask) == critical_mask
                    # And B is not in sol.
                    
                    # Iterate through all blocks? Too slow.
                    # We can precompute for each t-set, which blocks cover it?
                    # That's the `tset_to_blocks` from before, which was slow to build.
                    # But we can build it once here?
                    # Or just scan all blocks?
                    # 75k blocks * 10 tries * 50 iterations = 37M scans.
                    # Each scan: `if (block_masks[i] & critical_mask) == critical_mask`.
                    # This is fast.
                    
                    for i in range(num_blocks):
                        if i in sol: # O(N) check. Bad.
                            continue
                        # Better: maintain a set of selected indices
                        # Let's use a set `sol_set`
                        pass
                    
                    # Let's optimize this.
                    # We'll maintain a set of selected block indices.
                    # Re-initialize sol_set if not present.
                    # Actually, let's just use a boolean array `in_sol`.
                    
                    # To make this fast, we should precompute `tset_to_blocks` as a list of lists.
                    # Building it takes ~4.2M ops.
                    # Let's do it once in the main scope? No, inside local_search is fine if it's fast.
                    # Or pass it in.
                    
                    # Let's assume we have `tset_to_blocks` available.
                    # If not, we can compute it.
                    
                    # For now, let's just scan all blocks.
                    # We need to know if a block is in sol.
                    # `sol` is a list of indices.
                    # Converting to set every time is O(N).
                    # Let's maintain `in_sol` array.
                    
                    # I will refactor local_search to use `in_sol` array.
                    
                    found_swap = False
                    # ... (logic below)
                    
                if found_swap:
                    no_improve = 0
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    continue
            else:
                break
                
            no_improve += 1
            
        return best_sol

    # Refactored Local Search with precomputed tset_to_blocks
    # Build tset_to_blocks once
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tset_to_blocks[idx].append(b_idx)
            m ^= lsb
            
    def local_search_v2(initial_sol, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        in_sol = [False] * num_blocks
        for b in sol:
            in_sol[b] = True
            
        cover_counts = [0] * num_tsets
        
        def update_counts(block_idx, delta):
            m = block_masks[block_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_counts[idx] += delta
                m ^= lsb

        for b in sol:
            update_counts(b, 1)
            
        best_sol = list(sol)
        best_size = len(sol)
        no_improve = 0
        max_no_improve = 50
        
        while elapsed() < deadline - 10 and no_improve < max_no_improve:
            improved = False
            
            # Phase 1: Remove redundant
            order = list(range(len(sol)))
            rng.shuffle(order)
            for pos in order:
                if pos >= len(sol):
                    continue
                b = sol[pos]
                m = block_masks[b]
                redundant = True
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                    
                if redundant:
                    update_counts(b, -1)
                    in_sol[b] = False
                    sol.pop(pos)
                    improved = True
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    break
                    
            if improved:
                no_improve = 0
                continue
                
            # Phase 2: 1-for-1 Swap
            if sol:
                found_swap = False
                # Try more swaps for better improvement
                for _ in range(50):
                    if not sol:
                        break
                    pos = rng.randint(0, len(sol) - 1)
                    b_out = sol[pos]
                    
                    critical_mask = 0
                    m = block_masks[b_out]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] == 1:
                            critical_mask |= lsb
                        m ^= lsb
                        
                    if critical_mask == 0:
                        # Redundant
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        sol.pop(pos)
                        improved = True
                        found_swap = True
                        break
                        
                    # Get list of tsets in critical_mask
                    crit_ts_list = []
                    m = critical_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        crit_ts_list.append(idx)
                        m ^= lsb
                        
                    if not crit_ts_list:
                        continue
                        
                    # Start with candidates of first tset
                    candidates = set(tset_to_blocks[crit_ts_list[0]])
                    # Intersect with others
                    for ts_idx in crit_ts_list[1:]:
                        if not candidates:
                            break
                        candidates &= set(tset_to_blocks[ts_idx])
                        
                    # Remove blocks already in sol
                    sol_set = set(sol)
                    candidates -= sol_set
                    
                    if candidates:
                        b_in = rng.choice(list(candidates))
                        # Perform swap
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        update_counts(b_in, 1)
                        in_sol[b_in] = True
                        sol[pos] = b_in
                        found_swap = True
                        improved = True
                        break
                        
                if found_swap:
                    no_improve = 0
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    continue
            else:
                break
                
            no_improve += 1
            
        return best_sol

    # Main Loop
    best_solution = None
    best_size = float('inf')
    
    # Run more greedy restarts with the fixed algorithm
    for seed in range(10):
        if elapsed() > deadline - 15:
            break
        sol = greedy_cover(seed)
        if sol:
            ls_sol = local_search_v2(sol, seed)
            if len(ls_sol) < best_size:
                best_size = len(ls_sol)
                best_solution = ls_sol
                
    if best_solution is None:
        # Fallback: return empty or raise
        return []
        
    return [list(all_blocks[i]) for i in best_solution]
