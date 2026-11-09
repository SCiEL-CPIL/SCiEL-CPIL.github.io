"""Optimized construction of large 4-uniform sunflower-free family."""

def construct():
    from itertools import combinations
    import time
    import random
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Generate all 4-subsets of [1,16]
    all_sets = list(combinations(range(1, 17), 4))
    n_total = len(all_sets)
    
    # Convert to bitmasks for fast operations
    def to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m
    
    masks = [to_mask(s) for s in all_sets]
    
    # Precompute pairwise intersections between all sets
    # inter[i][j] = masks[i] & masks[j]
    inter = [[0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        mi = masks[i]
        for j in range(i + 1, n_total):
            v = mi & masks[j]
            inter[i][j] = v
            inter[j][i] = v
            
    # Optimize can_add by using local references
    def can_add(c_idx, family_indices, inter_ref=inter, masks_ref=masks):
        cm = masks_ref[c_idx]
        
        # Group family members by their intersection with c
        groups = {}
        for a_idx in family_indices:
            v = inter_ref[a_idx][c_idx]
            if v not in groups:
                groups[v] = []
            groups[v].append(a_idx)
        
        # For each group G_v, check if any pair (a,b) in G_v has inter[a][b] == v
        for v, group in groups.items():
            n_group = len(group)
            if n_group < 2:
                continue
            # Check all pairs in this group
            for i in range(n_group):
                a = group[i]
                for j in range(i + 1, n_group):
                    b = group[j]
                    if inter_ref[a][b] == v:
                        return False
        return True

    best_family = []
    best_size = 0
    
    orderings = []
    
    # Natural order
    orderings.append(list(range(n_total)))
    
    # Random orderings
    rng = random.Random(42)
    for _ in range(5):
        perm = list(range(n_total))
        rng.shuffle(perm)
        orderings.append(perm)
    
    # Heuristic: order by "popularity" - sets that intersect many others in small ways first
    avg_inter = [0] * n_total
    for i in range(n_total):
        total = 0
        for j in range(n_total):
            if i != j:
                total += bin(inter[i][j]).count('1')
        avg_inter[i] = total / (n_total - 1)
    
    # Sort by average intersection size (smaller first)
    by_avg = sorted(range(n_total), key=lambda x: avg_inter[x])
    orderings.append(by_avg)
    
    # Sort by reverse
    orderings.append(list(reversed(range(n_total))))
    
    for order in orderings:
        if time.time() > deadline - 10:
            break
        
        family = []
        family_set = set()
        
        for c_idx in order:
            if time.time() > deadline - 10:
                break
            if c_idx in family_set:
                continue
            
            if can_add(c_idx, family):
                family.append(c_idx)
                family_set.add(c_idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
    
    # Local search: Simple add-if-possible loop
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for c_idx in range(n_total):
                if time.time() > deadline - 5:
                    break
                if c_idx in set(best_family):
                    continue
                if can_add(c_idx, best_family):
                    best_family.append(c_idx)
                    improved = True
                    break

    # Increase number of restarts to explore more of the search space
    # The check is fast enough to allow many more restarts
    additional_restarts = 200
    for _ in range(additional_restarts):
        if time.time() > deadline - 10:
            break
        
        perm = list(range(n_total))
        rng.shuffle(perm)
        
        family = []
        family_set = set()
        
        for c_idx in perm:
            if time.time() > deadline - 10:
                break
            if c_idx in family_set:
                continue
            
            if can_add(c_idx, family):
                family.append(c_idx)
                family_set.add(c_idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
            
            # If we found a better solution, do a quick polish
            if time.time() < deadline - 5:
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    for c_idx in range(n_total):
                        if time.time() > deadline - 5:
                            break
                        if c_idx in set(best_family):
                            continue
                        if can_add(c_idx, best_family):
                            best_family.append(c_idx)
                            improved = True
                            break

    return [sorted(all_sets[i]) for i in best_family]
