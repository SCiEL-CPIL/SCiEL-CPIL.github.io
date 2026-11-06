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
    
    # Precompute partners[i][j] as bitmask of k such that {i, j, k} is a sunflower
    partner_cache = {}
    
    def get_partners(i, j):
        if i == j:
            return 0
        if i > j:
            i, j = j, i
        key = (i, j)
        if key in partner_cache:
            return partner_cache[key]
            
        si = all_sets[i]
        sj = all_sets[j]
        core = si & sj
        core_size = bin(core).count('1')
        
        # Union of si and sj
        union = si | sj
        U_mask = ((1 << 16) - 1) ^ union
        d_size = 4 - core_size
        
        partners_mask = 0
        if d_size < 0 or d_size > bin(U_mask).count('1'):
            partner_cache[key] = 0
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
                    partners_mask |= (1 << k)
                    
        partner_cache[key] = partners_mask
        return partners_mask

    # Initial Greedy Solution
    best_family = []
    best_size = 0
    
    start_time = time.time()
    timeout = 55
    
    for trial in range(30):
        if time.time() - start_time > timeout:
            break
        random.seed(trial * 12345 + 67890)
        indices = list(range(n))
        random.shuffle(indices)
        
        family = []
        family_indices = []
        candidates = (1 << n) - 1
        family_mask = 0
        
        for idx in indices:
            if not (candidates & (1 << idx)):
                continue
            
            # Check compatibility and calculate removals
            removed_mask = 0
            compatible = True
            cm = family_mask
            while cm:
                b = cm & -cm
                fi = b.bit_length() - 1
                p = get_partners(idx, fi)
                removed_mask |= p
                cm ^= b
                
            if removed_mask & family_mask:
                continue # Incompatible
            
            # Compatible. Add it.
            family_indices.append(idx)
            family_mask |= (1 << idx)
            candidates &= ~removed_mask
            candidates &= ~(1 << idx)
            family.append(all_sets[idx])
            
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
    best_indices = [mask_to_idx[m] for m in best_family]
    best_mask = 0
    for idx in best_indices:
        best_mask |= (1 << idx)
        
    # Branch and Bound
    def search(candidates_mask, current_mask, current_size):
        nonlocal best_mask, best_size
        
        # Upper bound
        if current_size + bin(candidates_mask).count('1') <= best_size:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Heuristic: Pick pivot with minimum loss (fewest candidates removed)
        # Get candidates list
        cands = []
        cm = candidates_mask
        while cm:
            b = cm & -cm
            idx = b.bit_length() - 1
            cands.append(idx)
            cm ^= b
            
        if not cands:
            return
            
        # If few candidates, evaluate all to find min loss
        # If many, just pick first (or random) to save time
        pivot = cands[0]
        min_loss = float('inf')
        
        if len(cands) <= 50:
            for x in cands:
                removed = 0
                cm_cur = current_mask
                while cm_cur:
                    b = cm_cur & -cm_cur
                    idx = b.bit_length() - 1
                    removed |= get_partners(x, idx)
                    cm_cur ^= b
                loss = bin(candidates_mask & removed).count('1')
                if loss < min_loss:
                    min_loss = loss
                    pivot = x
        else:
            # Random sample or first
            pivot = cands[0]

        # Branch 1: Include pivot
        removed_mask = 0
        cm_cur = current_mask
        while cm_cur:
            b = cm_cur & -cm_cur
            idx = b.bit_length() - 1
            removed_mask |= get_partners(pivot, idx)
            cm_cur ^= b
            
        new_candidates = candidates_mask & ~removed_mask & ~(1 << pivot)
        
        search(new_candidates, current_mask | (1 << pivot), current_size + 1)
        
        # Branch 2: Exclude pivot
        search(candidates_mask & ~(1 << pivot), current_mask, current_size)
        
        if time.time() - start_time > timeout:
            raise TimeoutError()

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
