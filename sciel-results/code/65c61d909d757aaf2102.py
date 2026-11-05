def construct():
    import time
    from itertools import combinations
    
    # 1. Setup
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    n = len(all_sets) # 1820
    mask_to_idx = {mask: i for i, mask in enumerate(all_sets)}
    
    # Precompute intersections for speed? 
    # We need to generate conflicts.
    
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
        
        # Union of si and sj
        union = si | sj
        # Elements in union but not in core
        # We need D subset of U = [16] \ union
        # such that |D| = 4 - core_size
        # And S_k = core | D
        # S_k must be distinct from si, sj
        
        U_mask = ((1 << 16) - 1) ^ union
        d_size = 4 - core_size
        
        conflicts_mask = 0
        if d_size < 0 or d_size > bin(U_mask).count('1'):
            conflict_cache[key] = 0
            return 0
            
        # Generate all subsets of U_mask of size d_size
        # U_mask has bits set for elements not in si or sj
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

    # Initial Greedy Solution
    # Use random shuffles to find a good start
    import random
    best_family = []
    best_size = 0
    
    start_time = time.time()
    timeout = 50
    
    for trial in range(50):
        if time.time() - start_time > timeout:
            break
        random.seed(trial * 12345 + 67890)
        indices = list(range(n))
        random.shuffle(indices)
        
        family = []
        family_indices = []
        
        for idx in indices:
            C = all_sets[idx]
            can_add = True
            
            for i in range(len(family_indices)):
                ai = family_indices[i]
                for j in range(i+1, len(family_indices)):
                    bj = family_indices[j]
                    # Check if ai, bj, idx form sunflower
                    # inter(ai, bj) == inter(ai, idx) == inter(bj, idx)
                    if (all_sets[ai] & all_sets[bj]) == (all_sets[ai] & C) == (all_sets[bj] & C):
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                family.append(C)
                family_indices.append(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
    # Convert best_family to indices
    best_indices = [mask_to_idx[m] for m in best_family]
    best_mask = 0
    for idx in best_indices:
        best_mask |= (1 << idx)
        
    # Branch and Bound
    # State: candidates_mask, current_mask (indices in current solution), current_size
    # We use the best found so far to prune
    
    def search(candidates_mask, current_mask, current_size):
        nonlocal best_mask, best_size
        
        # Upper bound: current_size + popcount(candidates_mask)
        if current_size + bin(candidates_mask).count('1') <= best_size:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Heuristic: Pick a vertex
        # Find the first set bit
        # Or pick the one with highest "degree" (conflicts with most other candidates via current set)
        # For speed, let's just pick the lowest index set bit
        # v_idx = (candidates_mask & -candidates_mask).bit_length() - 1
        
        # Better heuristic: Pick the vertex that blocks the most other candidates
        # This requires calculating conflicts with current set for each candidate, which is expensive.
        # Let's stick to first bit for now, or random?
        # Random might be better for diversity?
        # Let's try picking the one with the smallest index to be deterministic and fast.
        
        v_bit = candidates_mask & -candidates_mask
        v_idx = v_bit.bit_length() - 1
        
        # Branch 1: Include v_idx
        # Check if v_idx is compatible with current set
        # It should be, because it's in candidates_mask.
        # Update candidates
        new_candidates = candidates_mask & ~v_bit
        
        # Remove candidates that form sunflower with v_idx and any u in current_mask
        # Iterate through bits in current_mask
        cm = current_mask
        while cm:
            u_bit = cm & -cm
            u_idx = u_bit.bit_length() - 1
            cm ^= u_bit
            
            conf = get_conflicts(v_idx, u_idx)
            new_candidates &= ~conf
            
        # Recurse
        search(new_candidates, current_mask | v_bit, current_size + 1)
        
        # Branch 2: Exclude v_idx
        search(candidates_mask & ~v_bit, current_mask, current_size)
        
        if time.time() - start_time > 55:
            raise TimeoutError()

    # Initialize search with all sets as candidates
    # But we must ensure the initial current set is empty?
    # If we start with empty current set, candidates is all sets.
    # This will explore the whole space.
    # We can start with the best greedy solution as the "current" and try to add more?
    # No, B&B usually starts from scratch.
    # To use the greedy solution, we can just use it as the initial lower bound.
    # The search will find if there's a larger one.
    
    # However, searching from scratch with 1820 nodes might be slow.
    # We can restrict the search to sets that are "compatible" with the greedy solution?
    # No, the optimal solution might not contain the greedy solution.
    
    # Let's run the search.
    # To speed up, we can order the vertices.
    # The `search` function above picks the lowest index.
    
    # We need to handle the timeout gracefully.
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
