import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # LCM of 1..22
    L = 1
    for i in range(1, 23):
        L = math.lcm(L, i)
    
    weights = [0] * 23
    for i in range(1, 23):
        weights[i] = L // i
        
    # Check if a list of indices forms a dissociated set
    def is_dissociated(indices):
        if not indices:
            return True
        seen = {0}
        sums = [0]
        for idx in indices:
            w = weights[idx]
            # Optimization: check collisions before extending
            # new_sums = [s + w for s in sums]
            # for ns in new_sums:
            #     if ns in seen: return False
            #     seen.add(ns)
            # sums.extend(new_sums)
            
            # This is slightly faster in some cases?
            # Let's stick to the clear version.
            new_sums = []
            for s in sums:
                ns = s + w
                if ns in seen:
                    return False
                new_sums.append(ns)
                seen.add(ns)
            sums.extend(new_sums)
        return True

    best_set = []
    best_size = 0
    
    def update_best(s):
        nonlocal best_set, best_size
        if len(s) > best_size:
            best_set = s
            best_size = len(s)

    # 1. Greedy Ascending
    s = []
    for i in range(1, 23):
        if is_dissociated(s + [i]):
            s.append(i)
    update_best(s)
    
    # 2. Greedy Descending
    s = []
    for i in range(22, 0, -1):
        if is_dissociated(s + [i]):
            s.append(i)
    update_best(s)

    # 3. Random Restarts
    # Adaptive number of restarts based on time
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        if restarts > 2000: # Hard cap to prevent infinite loop if time check fails
            break
        elements = list(range(1, 23))
        random.shuffle(elements)
        s = []
        for i in elements:
            if is_dissociated(s + [i]):
                s.append(i)
        update_best(s)
        
    # 4. Local Search / Expansion on best found
    if best_set:
        current = set(best_set)
        # Try to add any missing element
        for i in range(1, 23):
            if i not in current:
                new_s = list(current | {i})
                if is_dissociated(new_s):
                    current.add(i)
                    update_best(list(current))
                    
    if not best_set:
        return [1]
        
    return best_set
