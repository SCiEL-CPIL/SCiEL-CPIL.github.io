import random
import time

def construct():
    prescribed = [1, 2, 5, 11]
    prescribed.sort()
    
    # Pre-compute sums for prescribed
    pres_sums = set()
    for i in range(len(prescribed)):
        for j in range(i, len(prescribed)):
            pres_sums.add(prescribed[i] + prescribed[j])
            
    best_set = list(prescribed)
    best_size = len(prescribed)
    
    start_time = time.time()
    time_limit = 110.0
    
    def is_valid_addition(x, current_set, current_sums):
        if 2 * x in current_sums:
            return False
        for y in current_set:
            if (x + y) in current_sums:
                return False
        return True

    # Phase 1: Randomized Greedy Restarts
    candidates = list(range(1, 285))
    
    while time.time() - start_time < time_limit - 5.0:
        current_set = list(prescribed)
        current_sums = set(pres_sums)
        
        # Shuffle candidates
        random.shuffle(candidates)
        
        for x in candidates:
            if x in current_set:
                continue
            if is_valid_addition(x, current_set, current_sums):
                current_set.append(x)
                for y in current_set[:-1]:
                    current_sums.add(x + y)
                current_sums.add(2 * x)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        if best_size == 20:
            break
            
    # Phase 2: Local Search / Swap improvement
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 1.0 and best_size < 20:
        current_set = list(best_set)
        current_sums = set()
        for i in range(len(current_set)):
            for j in range(i, len(current_set)):
                current_sums.add(current_set[i] + current_set[j])
                
        while time.time() - start_time < time_limit - 0.5:
            added = False
            for x in range(1, 285):
                if x in current_set:
                    continue
                if is_valid_addition(x, current_set, current_sums):
                    current_set.append(x)
                    for y in current_set[:-1]:
                        current_sums.add(x + y)
                    current_sums.add(2 * x)
                    added = True
                    break
            
            if added:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                    if best_size == 20:
                        break
                continue
            
            non_pres = [x for x in current_set if x not in prescribed]
            if not non_pres:
                break
            
            x_to_remove = random.choice(non_pres)
            candidates_not_in_set = [x for x in range(1, 285) if x not in current_set]
            if not candidates_not_in_set:
                break
                
            y_to_add = random.choice(candidates_not_in_set)
            
            new_set = [x for x in current_set if x != x_to_remove]
            new_set.append(y_to_add)
            new_set.sort()
            
            valid = True
            seen_sums = set()
            for i in range(len(new_set)):
                for j in range(i, len(new_set)):
                    s = new_set[i] + new_set[j]
                    if s in seen_sums:
                        valid = False
                        break
                    seen_sums.add(s)
                if not valid:
                    break
            
            if valid:
                current_set = new_set
                current_sums = seen_sums

    return best_set
