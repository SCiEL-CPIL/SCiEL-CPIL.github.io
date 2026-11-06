import random
import math
from itertools import combinations

def construct():
    # Generate all 4-element subsets of [1, 16] as bitmasks
    all_sets = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_sets = len(all_sets)
    
    # Precompute intersection table
    # inter[i][j] = all_sets[i] & all_sets[j]
    # 1820 x 1820 matrix of ints.
    # To save memory and time, we can just use bitwise AND on the fly if it's faster than memory access?
    # In Python, list indexing + bitwise AND vs memory lookup.
    # Let's try precomputing for the inner loop speed.
    # But building 3.3M entries takes time.
    # Let's see: 1820*1820 = 3,312,400.
    # Building takes ~0.1-0.5s. Acceptable.
    
    inter = [[0]*n_sets for _ in range(n_sets)]
    for i in range(n_sets):
        ai = all_sets[i]
        for j in range(i+1, n_sets):
            aj = all_sets[j]
            val = ai & aj
            inter[i][j] = val
            inter[j][i] = val
            
    # Helper to check if adding mask c to family F (list of indices) is valid
    # F contains indices into all_sets
    def is_valid_add(c_idx, F):
        c = all_sets[c_idx]
        for i in range(len(F)):
            ai = all_sets[F[i]]
            ac = ai & c
            for j in range(i+1, len(F)):
                bj = all_sets[F[j]]
                ab = inter[F[i]][F[j]]
                bc = bj & c
                if ab == ac == bc:
                    return False
        return True

    # Simulated Annealing
    best_family = []
    best_size = 0
    
    # Initialize with a greedy family
    random.seed(42)
    order = list(range(n_sets))
    random.shuffle(order)
    current_family = []
    current_set = set()
    
    for idx in order:
        if is_valid_add(idx, current_family):
            current_family.append(idx)
            current_set.add(idx)
            
    best_size = len(current_family)
    best_family = current_family.copy()
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.1
    alpha = 0.995
    num_steps = 5000
    T = T_start
    
    for step in range(num_steps):
        # Choose a move
        move_type = random.random()
        new_family = current_family[:]
        new_set = current_set.copy()
        
        if move_type < 0.4:
            # Add a random element
            # Pick random candidate from all_sets not in current_set
            # To avoid O(N) scan, we can pick random index and check
            # If in set, try again (few times)
            added = False
            for _ in range(10):
                cand = random.randint(0, n_sets-1)
                if cand not in new_set:
                    if is_valid_add(cand, new_family):
                        new_family.append(cand)
                        new_set.add(cand)
                        added = True
                    break
            if not added:
                # If we couldn't add, maybe try to remove and add (swap)
                # Or just skip this step
                pass
        elif move_type < 0.7:
            # Remove a random element
            if new_family:
                idx_out = random.randint(0, len(new_family)-1)
                s_out = new_family[idx_out]
                new_family.pop(idx_out)
                new_set.remove(s_out)
        else:
            # Swap
            if new_family:
                idx_out = random.randint(0, len(new_family)-1)
                s_out = new_family[idx_out]
                
                # Pick random candidate
                cand = random.randint(0, n_sets-1)
                if cand not in new_set:
                    # Check validity of adding cand to (new_family - s_out)
                    temp_family = [x for i, x in enumerate(new_family) if i != idx_out]
                    if is_valid_add(cand, temp_family):
                        new_family[idx_out] = cand
                        new_set.remove(s_out)
                        new_set.add(cand)
        
        # Evaluate
        new_size = len(new_family)
        if new_size >= best_size:
            best_size = new_size
            best_family = new_family.copy()
            
        # Acceptance
        delta = new_size - len(current_family)
        if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
            current_family = new_family
            current_set = new_set
            
        T *= alpha
        
    # Convert result back to list of lists
    result = []
    for mask_idx in best_family:
        mask = all_sets[mask_idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
