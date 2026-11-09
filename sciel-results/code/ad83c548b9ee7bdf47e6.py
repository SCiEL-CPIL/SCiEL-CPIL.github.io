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
    
    # For each set c, and each set a, compute:
    #   forbidden_mask[a][c] = set of b indices where a,b,c form a sunflower
    # But that's too much memory. Instead, we'll compute on the fly.
    
    # Key insight: For candidate c and existing family F:
    # c conflicts with pair (a,b) in F if inter[a][b] == inter[a][c] == inter[b][c]
    
    # Efficient check: for candidate c, group existing sets by their intersection with c.
    # For each intersection value v, let G_v = {a in F : inter[a][c] == v}
    # A conflict exists if there are a,b in G_v (a != b) such that inter[a][b] == v
    
    # So for each v, we need to check if any pair in G_v has intersection v.
    
    def can_add(c_idx, family_indices):
        """Check if set c_idx can be added to family_indices without creating a sunflower."""
        cm = masks[c_idx]
        
        # Group family members by their intersection with c
        groups = {}
        for a_idx in family_indices:
            v = inter[a_idx][c_idx]
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
                    if inter[a][b] == v:
                        return False
        return True
    
    # Greedy construction with multiple orderings
    best_family = []
    best_size = 0
    
    # Try several different orderings
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
    # Actually, let's try: sets with smaller average intersection size with others
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
    
    # Local search: try to improve by swapping
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_size = len(best_family)
            
            # Try to add a set
            for c_idx in range(n_total):
                if time.time() > deadline - 5:
                    break
                if c_idx in set(best_family):
                    continue
                if can_add(c_idx, best_family):
                    best_family.append(c_idx)
                    improved = True
                    break
            
            # Try swaps: remove one, try to add two
            if time.time() < deadline - 5:
                for pos in range(len(best_family)):
                    if time.time() > deadline - 5:
                        break
                    removed = best_family[pos]
                    temp_family = [best_family[i] for i in range(len(best_family)) if i != pos]
                    temp_set = set(temp_family)
                    
                    # Try to find two sets that can be added
                    found_two = False
                    for c1 in range(n_total):
                        if time.time() > deadline - 5:
                            break
                        if c1 in temp_set or c1 == removed:
                            continue
                        if can_add(c1, temp_family):
                            new_family_1 = temp_family + [c1]
                            for c2 in range(n_total):
                                if time.time() > deadline - 5:
                                    break
                                if c2 in temp_set or c2 == removed or c2 == c1:
                                    continue
                                if can_add(c2, new_family_1):
                                    best_family = new_family_1 + [c2]
                                    improved = True
                                    found_two = True
                                    break
                        if found_two:
                            break
                    if found_two:
                        break
    
    return [sorted(all_sets[i]) for i in best_family]
