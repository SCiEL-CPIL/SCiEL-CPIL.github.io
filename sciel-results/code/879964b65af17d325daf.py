import random
from itertools import combinations

def construct():
    # Precompute all 4-subsets of [16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n = len(all_sets)  # 1820
    
    best_size = 0
    best_family = []
    
    # Number of trials to fit in time
    # Each trial involves greedy selection.
    # Greedy selection is O(N * |F| * avg_group_cost).
    # |F| ~ 200. N = 1820.
    # 1820 * 200 = 364,000 operations per trial roughly.
    # 500 trials -> 180 million operations. Might be too slow in Python.
    # Let's aim for 100-200 trials and optimize the inner loop.
    
    for trial in range(200):
        random.seed(trial)
        indices = list(range(n))
        random.shuffle(indices)
        
        chosen = []
        # To speed up, we can maintain a list of chosen masks
        # and maybe a set for O(1) lookup if needed?
        
        for idx in indices:
            c = all_sets[idx]
            conflict = False
            
            # Group chosen sets by their intersection with c
            # buckets: dict {intersection_mask: list_of_chosen_masks}
            buckets = {}
            
            for a in chosen:
                inter = a & c
                if inter not in buckets:
                    buckets[inter] = []
                buckets[inter].append(a)
            
            # Check for conflicts in each bucket
            for y, group in buckets.items():
                if len(group) < 2:
                    continue
                # Check if any two in group are disjoint outside y
                g_len = len(group)
                # Optimization: if g_len is small, O(g^2) is fine
                for i in range(g_len):
                    ai = group[i]
                    ai_out = ai & ~y
                    for j in range(i + 1, g_len):
                        aj = group[j]
                        aj_out = aj & ~y
                        if (ai_out & aj_out) == 0:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break
            
            if not conflict:
                chosen.append(c)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
            
            # Early stopping if we hit a very high bound?
            # Max is 250. If we get 240+, we can stop?
            # But we want to maximize.
            
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        elems = [i + 1 for i in range(16) if mask & (1 << i)]
        result.append(elems)
    
    return result
