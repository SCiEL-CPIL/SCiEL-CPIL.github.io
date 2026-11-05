import random
from itertools import combinations

def construct():
    all_sets = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_sets = len(all_sets)
    best_family = []
    best_size = 0
    
    # Precompute intersections for speed? 
    # inter[i][j] is a 16-bit int.
    # 1820 * 1820 * 4 bytes = ~13 MB. Acceptable.
    # But building it takes time.
    # Let's just use bitwise AND, it's very fast in Python.
    
    def is_valid_add(c, F):
        # F is list of masks
        for i in range(len(F)):
            a = F[i]
            ac = a & c
            for j in range(i+1, len(F)):
                b = F[j]
                if (a & b) == ac == (b & c):
                    return False
        return True

    for seed in range(20):
        random.seed(seed)
        order = list(range(n_sets))
        random.shuffle(order)
        
        family = []
        family_set = set()
        
        for idx in order:
            c = all_sets[idx]
            if is_valid_add(c, family):
                family.append(c)
                family_set.add(c)
        
        current_size = len(family)
        if current_size > best_size:
            best_size = current_size
            best_family = family.copy()
        
        # Local Search
        for _ in range(100):
            # Try to add
            added = False
            # Sample a few candidates from complement to save time
            # Or just iterate all, but break early
            for c in all_sets:
                if c not in family_set:
                    if is_valid_add(c, family):
                        family.append(c)
                        family_set.add(c)
                        current_size += 1
                        added = True
                        break
            
            if added:
                if current_size > best_size:
                    best_size = current_size
                    best_family = family.copy()
                continue
            
            # Try 5 random swaps
            swap_done = False
            for _ in range(5):
                if not family:
                    break
                idx_out = random.randint(0, len(family)-1)
                s_out = family[idx_out]
                
                # Pick random candidate from complement
                # To avoid O(N) scan, we can maintain a list of complement indices
                # But for simplicity, pick random index from all_sets and check if in family
                # This is biased but okay for heuristic
                c_in = all_sets[random.randint(0, n_sets-1)]
                if c_in in family_set:
                    continue
                
                new_family = [x for i, x in enumerate(family) if i != idx_out]
                if is_valid_add(c_in, new_family):
                    family[idx_out] = c_in
                    family_set.remove(s_out)
                    family_set.add(c_in)
                    swap_done = True
                    break
            
            if not swap_done and random.random() < 0.05:
                # Perturb
                if family:
                    idx_out = random.randint(0, len(family)-1)
                    s_out = family[idx_out]
                    family.pop(idx_out)
                    family_set.remove(s_out)
                    current_size -= 1

    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
