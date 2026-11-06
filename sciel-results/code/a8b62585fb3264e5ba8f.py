import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 16
    k = 4
    
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    num_sets = len(all_sets)
    full_mask = (1 << n) - 1
    
    best_family = []
    best_size = 0
    
    num_restarts = 500
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        order = list(range(num_sets))
        random.shuffle(order)
        
        family = []
        
        for idx in order:
            C = all_sets[idx]
            
            groups = {}
            for fi in family:
                A = all_sets[fi]
                inter = A & C
                if inter not in groups:
                    groups[inter] = []
                groups[inter].append(fi)
            
            valid = True
            for S, grp in groups.items():
                if len(grp) < 2:
                    continue
                notS = full_mask ^ S
                outsides = [all_sets[fi] & notS for fi in grp]
                m = len(outsides)
                for ii in range(m):
                    oi = outsides[ii]
                    for jj in range(ii+1, m):
                        if (oi & outsides[jj]) == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                family.append(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
    
    result = [sorted([x for x in range(n) if (all_sets[i] >> x) & 1]) for i in best_family]
    return result
