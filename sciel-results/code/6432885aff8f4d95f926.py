import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 16
    all_sets = []
    for c in combinations(range(n), 4):
        mask = 0
        for x in c:
            mask |= (1 << x)
        all_sets.append(mask)
        
    best_family = []
    
    # Run multiple restarts
    while time.time() < deadline:
        random.shuffle(all_sets)
        family = []
        
        for s in all_sets:
            valid = True
            # Optimization: if len(family) is large, we can break early if we find a conflict
            for i in range(len(family)):
                a = family[i]
                for j in range(i + 1, len(family)):
                    b = family[j]
                    if (a & b) == (a & s) == (b & s):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                family.append(s)
                
        if len(family) > len(best_family):
            best_family = family
            if len(best_family) >= 250:
                break

    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
        
    return result
