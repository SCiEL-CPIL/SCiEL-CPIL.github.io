import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 16
    k = 4
    all_sets = []
    for combo in itertools.combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
        
    num_sets = len(all_sets)
    
    def greedy_build(order):
        chosen = []
        for s in order:
            if len(chosen) < 2:
                chosen.append(s)
                continue
                
            groups = {}
            for i, a in enumerate(chosen):
                inter = a & s
                if inter not in groups:
                    groups[inter] = []
                groups[inter].append(i)
            
            valid = True
            for k_mask, indices in groups.items():
                if len(indices) < 2:
                    continue
                for i_idx in range(len(indices)):
                    i = indices[i_idx]
                    a = chosen[i]
                    for j_idx in range(i_idx + 1, len(indices)):
                        j = indices[j_idx]
                        b = chosen[j]
                        if (a & b) == k_mask:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                    
            if valid:
                chosen.append(s)
                
        return chosen

    best_size = 0
    best_family = []
    
    # Run multiple random trials
    while time.time() < deadline:
        order = all_sets[:]
        random.shuffle(order)
        family = greedy_build(order)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
    result = []
    for mask in best_family:
        subset = [i for i in range(n) if (mask >> i) & 1]
        result.append(sorted(subset))
        
    return result
