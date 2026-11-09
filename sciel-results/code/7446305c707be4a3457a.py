"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    total = len(all_sets)  # 1820
    
    def can_add(chosen_indices, new_mask):
        """Check if new_mask can be added given chosen_indices."""
        n_chosen = len(chosen_indices)
        for i in range(n_chosen):
            a = all_sets[chosen_indices[i]]
            a_inter_new = a & new_mask
            for j in range(i+1, n_chosen):
                b = all_sets[chosen_indices[j]]
                b_inter_new = b & new_mask
                a_inter_b = a & b
                if a_inter_b == a_inter_new and a_inter_new == b_inter_new:
                    return False
        return True
    
    best_size = 0
    best_family = []
    
    # Try multiple random orderings with different seeds
    rng = random.Random(42)
    
    for restart in range(100):
        if time.time() > deadline - 5:
            break
        
        order = list(range(total))
        rng.shuffle(order)
        
        chosen_indices = []
        
        for idx in order:
            if time.time() > deadline - 2:
                break
            mask = all_sets[idx]
            if can_add(chosen_indices, mask):
                chosen_indices.append(idx)
        
        if len(chosen_indices) > best_size:
            best_size = len(chosen_indices)
            best_family = chosen_indices[:]
    
    # Convert bitmasks back to sorted lists
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elems = [i+1 for i in range(n) if mask & (1 << i)]
        result.append(elems)
    
    return result
# EVOLVE-BLOCK-END
