"""Maximum 4-uniform family on [16] with no three-petal sunflowers (including empty core)."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 80
    
    n = 16
    k = 4
    all_sets = list(combinations(range(n), k))
    m = len(all_sets)  # 1820
    
    # Convert to bitmasks
    masks = [0] * m
    for i, s in enumerate(all_sets):
        mask = 0
        for x in s:
            mask |= (1 << x)
        masks[i] = mask
    
    # Precompute: for each pair (i,j), store the core = masks[i]&masks[j]
    # And for each core value, store which pairs have that core.
    # A new set k conflicts with pair (i,j) if:
    #   masks[k]&masks[i] == core_ij AND masks[k]&masks[j] == core_ij
    
    # Better: for each set k, precompute for each pair (i,j) whether k conflicts.
    # That's still O(m^3).
    
    # Alternative efficient check:
    # When adding set k, for each existing set i, compute ci = masks[k]&masks[i].
    # Then for each existing set j (j>i), check if masks[k]&masks[j] == ci AND masks[i]&masks[j] == ci.
    # We can precompute all pair intersections in a 2D array.
    
    # m = 1820, so m^2 = 3.3M. Store as a list of lists of ints.
    # pair_inter[i][j] = masks[i] & masks[j] for i < j
    # Use a flat array: pair_inter[i*m + j] for i < j
    
    pair_inter = [0] * (m * m)
    for i in range(m):
        mi = masks[i]
        for j in range(i+1, m):
            pair_inter[i*m + j] = mi & masks[j]
            pair_inter[j*m + i] = pair_inter[i*m + j]
    
    def check_add(chosen_indices, new_idx):
        """Check if new_idx can be added to chosen_indices without sunflower."""
        new_mask = masks[new_idx]
        L = len(chosen_indices)
        # For each pair (a, b) in chosen_indices, check if (a, b, new_idx) is a sunflower
        for ai in range(L):
            a = chosen_indices[ai]
            ma = masks[a]
            inter_an = ma & new_mask
            for bi in range(ai+1, L):
                b = chosen_indices[bi]
                mb = masks[b]
                # Sunflower condition: ma&mb == ma&new_mask == mb&new_mask
                if pair_inter[a*m + b] == inter_an and mb & new_mask == inter_an:
                    return False
        return True
    
    def greedy_build(order):
        chosen = []
        for idx in order:
            if check_add(chosen, idx):
                chosen.append(idx)
        return chosen
    
    all_indices = list(range(m))
    
    # Try natural order first
    result = greedy_build(all_indices)
    best_size = len(result)
    best_family = [all_sets[i] for i in result]
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return [list(s) for s in best_family]
    
    # Try random orders
    random.seed(42)
    for trial in range(50):
        if time.time() - start_time > DEADLINE - 5:
            break
        random_order = all_indices[:]
        random.shuffle(random_order)
        result = greedy_build(random_order)
        if len(result) > best_size:
            best_size = len(result)
            best_family = [all_sets[i] for i in result]
    
    return [list(s) for s in best_family]
