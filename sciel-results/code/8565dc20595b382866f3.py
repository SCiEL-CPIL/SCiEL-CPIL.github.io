import random
from itertools import combinations

def construct():
    # Precompute all 4-subsets as bitmasks
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    # Precompute cores and outers for each set
    # core_outers[set_idx] = list of (core_mask, outer_mask)
    set_cores = {}
    for idx, s in enumerate(all_sets):
        cores = []
        # Iterate all submasks of s
        # s has 4 bits. Submasks: 16 total.
        # We want proper submasks (size 0,1,2,3)
        sub = s
        while True:
            if sub != s: # Proper subset
                cores.append((sub, s ^ sub))
            if sub == 0: break
            sub = (sub - 1) & s
        set_cores[idx] = cores

    # Helper to check if a list of outers contains 3 pairwise disjoint
    def has_three_disjoint(outers):
        n = len(outers)
        if n < 3:
            return None
        # Optimization for small outer sizes
        # If outer size is 1 (bitmask with 1 bit), check distinct count
        # If outer size is 2, check pairs
        # General check
        for i in range(n):
            oi = outers[i]
            for j in range(i+1, n):
                oj = outers[j]
                if oi & oj:
                    continue
                for k in range(j+1, n):
                    ok = outers[k]
                    if (oi & ok) == 0 and (oj & ok) == 0:
                        return (i, j, k)
        return None

    # Helper to find a sunflower in the current chosen list
    # Returns (idx_a, idx_b, idx_c) or None
    def find_sunflower(chosen_indices):
        # Build core_to_outers_map
        # core_mask -> list of (outer_mask, set_idx)
        core_map = {}
        for idx in chosen_indices:
            for core, outer in set_cores[idx]:
                if core not in core_map:
                    core_map[core] = []
                core_map[core].append((outer, idx))
        
        # Check each core
        for core, entries in core_map.items():
            if len(entries) < 3:
                continue
            outers = [e[0] for e in entries]
            # Check for 3 pairwise disjoint
            n = len(outers)
            # Optimization: if outer size is 1, just check distinct
            # Determine outer size from first entry
            # Actually, outer size is constant for a given core size
            # But we can just run the general check
            found = has_three_disjoint(outers)
            if found:
                i, j, k = found
                return (entries[i][1], entries[j][1], entries[k][1])
        return None

    # Main Logic
    best_family = []
    best_size = 0
    
    for restart in range(50):
        # Start with a random large set
        random.shuffle(all_sets)
        initial_size = 120
        chosen_indices = list(range(initial_size)) # Indices into all_sets
        chosen_set = set(chosen_indices)
        
        # Repair Phase
        # Limit iterations to avoid infinite loop
        for _ in range(1000):
            sun = find_sunflower(chosen_indices)
            if sun is None:
                break
            # Remove a random element from the sunflower
            remove_idx = random.choice(sun)
            chosen_indices.remove(remove_idx)
            chosen_set.discard(remove_idx)
        
        # Fill Phase
        # Try to add remaining sets
        remaining = [i for i in range(len(all_sets)) if i not in chosen_set]
        random.shuffle(remaining)
        
        for idx in remaining:
            # Check if adding idx creates a sunflower
            # Temporarily add
            new_indices = chosen_indices + [idx]
            sun = find_sunflower(new_indices)
            if sun is None:
                chosen_indices = new_indices
                chosen_set.add(idx)
            else:
                # If it creates a sunflower, we could try to repair by removing one?
                # For now, just skip to keep it simple and fast
                pass
        
        if len(chosen_indices) > best_size:
            best_size = len(chosen_indices)
            best_family = chosen_indices[:]
            
    # Convert back to lists
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result
