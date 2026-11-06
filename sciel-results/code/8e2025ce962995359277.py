import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0
    n = 4096
    max_sum = 3 * n
    best_set = []
    best_size = 0
    
    # Core set: Powers of 3
    core = []
    p = 1
    while p <= n:
        core.append(p)
        p *= 3
    
    trials = 0
    while time.time() - start_time < time_limit:
        trials += 1
        
        # Strategy 0: Random shuffle
        # Strategy 1: Ascending
        # Strategy 2: Descending
        # Strategy 3: Random, but start with core
        # Strategy 4: Random, but bias towards large numbers?
        
        strategy = trials % 5
        
        if strategy == 0:
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
            current_set = []
            sums_present = bytearray(max_sum + 1)
        elif strategy == 1:
            candidates = list(range(1, n + 1))
            current_set = []
            sums_present = bytearray(max_sum + 1)
        elif strategy == 2:
            candidates = list(range(n, 0, -1))
            current_set = []
            sums_present = bytearray(max_sum + 1)
        else:
            # Start with core
            current_set = core[:]
            sums_present = bytearray(max_sum + 1)
            for x in current_set:
                sums_present[3 * x] = 1
                for y in current_set:
                    sums_present[2 * x + y] = 1
                for i in range(len(current_set)):
                    y = current_set[i]
                    for j in range(i, len(current_set)):
                        z = current_set[j]
                        sums_present[x + y + z] = 1
            
            # Candidates are the rest
            core_set = set(core)
            candidates = [x for x in range(1, n + 1) if x not in core_set]
            random.shuffle(candidates)
            
        # Greedy extension
        for x in candidates:
            if time.time() - start_time > time_limit - 0.5:
                break
                
            # Check if x can be added
            valid = True
            
            if sums_present[3 * x]:
                valid = False
            else:
                for y in current_set:
                    if sums_present[2 * x + y]:
                        valid = False
                        break
                
                if valid:
                    len_s = len(current_set)
                    for i in range(len_s):
                        y = current_set[i]
                        for j in range(i, len_s):
                            z = current_set[j]
                            if sums_present[x + y + z]:
                                valid = False
                                break
                        if not valid:
                            break
            
            if valid:
                current_set.append(x)
                sums_present[3 * x] = 1
                for y in current_set:
                    sums_present[2 * x + y] = 1
                len_s = len(current_set)
                for i in range(len_s):
                    y = current_set[i]
                    for j in range(i, len_s):
                        z = current_set[j]
                        sums_present[x + y + z] = 1
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    return sorted(best_set)
