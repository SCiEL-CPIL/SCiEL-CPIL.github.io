def construct():
    import time
    from itertools import combinations
    import random

    # 1. Setup
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    n = len(all_sets) # 1820
    mask_to_idx = {mask: i for i, mask in enumerate(all_sets)}
    
    conflict_cache = {}
    
    def get_conflicts(i, j):
        if i > j:
            i, j = j, i
        key = (i, j)
        if key in conflict_cache:
            return conflict_cache[key]
        
        si = all_sets[i]
        sj = all_sets[j]
        core = si & sj
        core_size = bin(core).count('1')
        
        union = si | sj
        U_mask = ((1 << 16) - 1) ^ union
        d_size = 4 - core_size
        
        conflicts_mask = 0
        if d_size < 0 or d_size > bin(U_mask).count('1'):
            conflict_cache[key] = 0
            return 0
            
        u_elems = []
        for x in range(16):
            if U_mask & (1 << x):
                u_elems.append(x)
                
        for d_combo in combinations(u_elems, d_size):
            d_mask = 0
            for x in d_combo:
                d_mask |= (1 << x)
            sk = core | d_mask
            if sk in mask_to_idx:
                k = mask_to_idx[sk]
                if k != i and k != j:
                    conflicts_mask |= (1 << k)
                    
        conflict_cache[key] = conflicts_mask
        return conflicts_mask

    # Precompute static heuristic scores
    # Score(i) = sum_{j != i} popcount(get_conflicts(i, j))
    # This approximates how "conflict-prone" i is.
    # We want to pick high score first? Or low score?
    # In MIS, we often pick high degree first to prune.
    # Here, high score means it conflicts with many others.
    # Picking it early forces us to remove many others, which might be good for pruning the search space (branching on it eliminates many candidates).
    
    scores = [0] * n
    start_time = time.time()
    
    # Calculate scores
    # This is O(N^2 * cost). 
    # 1820^2 / 2 = 1.6M pairs.
    # If cost is ~100 ops, 160M ops. Might be slow in Python.
    # Let's limit this or use a faster approximation.
    # Maybe just count the number of sunflowers i is part of?
    # A sunflower is a triple {i, j, k}.
    # i is in a sunflower with pair (j, k) if k in conflicts(i, j).
    # So count(i) = sum_j popcount(conflicts(i, j)).
    
    # To speed up, we can sample or just use a simple degree.
    # Let's try to compute it but with a timeout.
    
    for i in range(n):
        if time.time() - start_time > 10:
            break
        total = 0
        for j in range(n):
            if i == j: continue
            total += bin(get_conflicts(i, j)).count('1')
        scores[i] = total
        
    # Sort indices by score descending
    sorted_indices = sorted(range(n), key=lambda x: scores[x], reverse=True)
    # Create a rank array for quick lookup? Or just iterate sorted_indices and check if in candidates.
    
    # Greedy Initial Solution
    best_family_mask = 0
    best_size = 0
    
    for trial in range(50):
        if time.time() - start_time > 20:
            break
        random.seed(trial)
        # Shuffle the sorted order? Or just use it?
        # Using the static order for greedy might be good.
        # Let's mix: use random order for some trials, static for others.
        if trial % 2 == 0:
            order = sorted_indices[:]
        else:
            order = list(range(n))
            random.shuffle(order)
            
        family_indices = []
        for idx in order:
            can_add = True
            for i in range(len(family_indices)):
                ai = family_indices[i]
                for j in range(i+1, len(family_indices)):
                    bj = family_indices[j]
                    if (all_sets[ai] & all_sets[bj]) == (all_sets[ai] & all_sets[idx]) == (all_sets[bj] & all_sets[idx]):
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                family_indices.append(idx)
        
        if len(family_indices) > best_size:
            best_size = len(family_indices)
            best_family_mask = 0
            for idx in family_indices:
                best_family_mask |= (1 << idx)

    # B&B
    best_mask = best_family_mask
    
    def search(candidates_mask, current_mask, current_size):
        nonlocal best_mask, best_size
        
        if current_size + bin(candidates_mask).count('1') <= best_size:
            return
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        if time.time() - start_time > 55:
            raise TimeoutError()

        # Pick the first available index in sorted_indices
        v_idx = -1
        for idx in sorted_indices:
            if candidates_mask & (1 << idx):
                v_idx = idx
                break
        
        if v_idx == -1:
            return
            
        v_bit = 1 << v_idx
        
        # Calculate new candidates
        new_candidates = candidates_mask & ~v_bit
        
        # Remove conflicts with current set
        cm = current_mask
        while cm:
            u_bit = cm & -cm
            u_idx = u_bit.bit_length() - 1
            cm ^= u_bit
            conf = get_conflicts(v_idx, u_idx)
            new_candidates &= ~conf
            
        # Branch 1: Include
        search(new_candidates, current_mask | v_bit, current_size + 1)
        
        # Branch 2: Exclude
        search(candidates_mask & ~v_bit, current_mask, current_size)

    try:
        all_candidates = (1 << n) - 1
        search(all_candidates, 0, 0)
    except TimeoutError:
        pass
    
    # Extract result
    result = []
    for idx in range(n):
        if best_mask & (1 << idx):
            mask = all_sets[idx]
            elems = []
            for x in range(16):
                if mask & (1 << x):
                    elems.append(x + 1)
            result.append(sorted(elems))
            
    return result
