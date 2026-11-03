import itertools
import time
import random
import math

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
        cover_counts = [0] * num_tsets
        
        def update_counts(block_idx, delta):
            mask = block_masks[block_idx]
            m = mask
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
            
            # Phase 1: Try to remove redundant blocks
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
                    update_counts(b, -1)
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
                        update_counts(b_out, -1)
                        sol.pop(pos)
                        improved = True
                        found_swap = True
                        break
                        
                    # Find a block covering critical_mask
                    # We need a block B such that (block_masks[B] & critical_mask) == critical_mask
                    # And B is not in sol.
                    
                    # Use tset_to_blocks intersection for speed
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
                        
                    candidates = set(tset_to_blocks[crit_ts_list[0]])
                    for ts_idx in crit_ts_list[1:]:
                        if not candidates:
                            break
                        candidates &= set(tset_to_blocks[ts_idx])
                        
                    sol_set = set(sol)
                    candidates -= sol_set
                    
                    if candidates:
                        b_in = rng.choice(list(candidates))
                        update_counts(b_out, -1)
                        update_counts(b_in, 1)
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

    # Build tset_to_blocks once
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tset_to_blocks[idx].append(b_idx)
            m ^= lsb
            
    # 4. Simulated Annealing
    def simulated_annealing(initial_sol, seed):
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
        
        # Cooling schedule
        T = 5.0
        cooling_rate = 0.99
        min_T = 0.1
        max_iterations = 10000
        
        for it in range(max_iterations):
            if elapsed() > deadline - 15:
                break
            if T < min_T:
                break
                
            # Generate a random move
            move_type = rng.random()
            
            if move_type < 0.3:
                # Add a random block
                if not sol:
                    continue
                # Pick a random block not in sol
                # Efficient way: pick random index, check if in sol
                # Or maintain a list of available blocks? Too heavy to update.
                # Just pick random index from all blocks.
                b_in = rng.randint(0, num_blocks - 1)
                if in_sol[b_in]:
                    continue
                # Apply
                update_counts(b_in, 1)
                in_sol[b_in] = True
                sol.append(b_in)
                new_size = len(sol)
                old_size = new_size - 1
                delta_e = new_size - old_size # +1
                if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                    if new_size < best_size:
                        best_size = new_size
                        best_sol = list(sol)
                else:
                    # Revert
                    update_counts(b_in, -1)
                    in_sol[b_in] = False
                    sol.pop()
                    
            elif move_type < 0.6:
                # Remove a random block
                if not sol:
                    continue
                pos = rng.randint(0, len(sol) - 1)
                b_out = sol[pos]
                # Check if removal is valid (all tsets still covered)
                # We can check cover_counts > 1 for all tsets in b_out
                valid = True
                m = block_masks[b_out]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        valid = False
                        break
                    m ^= lsb
                    
                if valid:
                    # Apply
                    update_counts(b_out, -1)
                    in_sol[b_out] = False
                    sol.pop(pos)
                    new_size = len(sol)
                    old_size = new_size + 1
                    delta_e = new_size - old_size # -1
                    # Always accept improvement
                    if new_size < best_size:
                        best_size = new_size
                        best_sol = list(sol)
                else:
                    # Revert (do nothing)
                    pass
                    
            else:
                # Swap
                if len(sol) < 2:
                    continue
                pos1 = rng.randint(0, len(sol) - 1)
                b_out = sol[pos1]
                
                # Find critical tsets for b_out
                critical_mask = 0
                m = block_masks[b_out]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        critical_mask |= lsb
                    m ^= lsb
                    
                if critical_mask == 0:
                    # Can just remove, handled in remove phase, but let's try swap anyway or remove
                    # If critical is 0, it's redundant.
                    # For simplicity in SA, if redundant, just remove.
                    update_counts(b_out, -1)
                    in_sol[b_out] = False
                    sol.pop(pos1)
                    new_size = len(sol)
                    old_size = new_size + 1
                    if new_size < best_size:
                        best_size = new_size
                        best_sol = list(sol)
                    continue

                # Find candidate b_in
                crit_ts_list = []
                m = critical_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    crit_ts_list.append(idx)
                    m ^= lsb
                    
                candidates = set(tset_to_blocks[crit_ts_list[0]])
                for ts_idx in crit_ts_list[1:]:
                    if not candidates:
                        break
                    candidates &= set(tset_to_blocks[ts_idx])
                    
                sol_set = set(sol)
                candidates -= sol_set
                
                if not candidates:
                    continue
                    
                b_in = rng.choice(list(candidates))
                
                # Apply swap
                update_counts(b_out, -1)
                in_sol[b_out] = False
                update_counts(b_in, 1)
                in_sol[b_in] = True
                sol[pos1] = b_in
                
                new_size = len(sol)
                old_size = new_size # Size unchanged
                delta_e = 0
                # Accept (always for size 0 delta, but we might want to explore)
                # Since size is same, we accept to explore the state space
                if new_size < best_size:
                    best_size = new_size
                    best_sol = list(sol)
                    
            T *= cooling_rate

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
            # First, standard local search to polish
            ls_sol = local_search(sol, seed)
            
            # Then, Simulated Annealing to escape local minima
            sa_sol = simulated_annealing(ls_sol, seed)
            
            if len(sa_sol) < best_size:
                best_size = len(sa_sol)
                best_solution = sa_sol
                
    if best_solution is None:
        return []
        
    return [list(all_blocks[i]) for i in best_solution]
