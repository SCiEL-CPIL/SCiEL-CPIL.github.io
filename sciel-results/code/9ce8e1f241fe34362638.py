import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    
    def greedy_build(order):
        current_set = list(prescribed)
        current_sums = set()
        for i, x in enumerate(current_set):
            for y in current_set[i:]:
                current_sums.add(x + y)
        
        for x in order:
            if x in current_set:
                continue
            new_sums = {x + y for y in current_set} | {2 * x}
            if len(new_sums) == len(current_set) + 1 and not (new_sums & current_sums):
                current_set.append(x)
                current_sums.update(new_sums)
        return current_set

    best_set = list(prescribed)
    best_size = len(best_set)
    
    s1 = greedy_build(range(12, 285))
    if len(s1) > best_size:
        best_set = s1
        best_size = len(best_set)
        
    rng = random.Random(42)
    for _ in range(20):
        order = list(range(1, 285))
        rng.shuffle(order)
        s2 = greedy_build(order)
        if len(s2) > best_size:
            best_set = s2
            best_size = len(best_set)

    all_candidates = [x for x in range(1, 285) if x not in prescribed_set]
    
    def greedy_fill_estimate(current_set, current_sums_mask, candidates):
        sim_set = list(current_set)
        sim_sums = current_sums_mask
        for c in candidates:
            new_sums_mask = 0
            valid = True
            for x in sim_set:
                s = c + x
                if (sim_sums >> s) & 1 or (new_sums_mask >> s) & 1:
                    valid = False
                    break
                new_sums_mask |= (1 << s)
            if valid:
                s2 = 2 * c
                if (sim_sums >> s2) & 1 or (new_sums_mask >> s2) & 1:
                    valid = False
                else:
                    new_sums_mask |= (1 << s2)
                    sim_set.append(c)
                    sim_sums = sim_sums | new_sums_mask
        return len(sim_set)

    def dfs(current_set, current_sums_mask, candidates, target_k):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return False
            
        if len(current_set) == target_k:
            if target_k > best_size:
                best_size = target_k
                best_set = list(current_set)
            return True
        
        if len(current_set) + len(candidates) < target_k:
            return False
            
        # Pruning with greedy estimate
        # Only do this if we are close to the target or deep in the tree to save time?
        # Or always, since it's fast.
        if greedy_fill_estimate(current_set, current_sums_mask, candidates) < target_k:
            return False

        for i, c in enumerate(candidates):
            new_sums_mask = 0
            valid = True
            
            for x in current_set:
                s = c + x
                if (current_sums_mask >> s) & 1 or (new_sums_mask >> s) & 1:
                    valid = False
                    break
                new_sums_mask |= (1 << s)
            
            if valid:
                s2 = 2 * c
                if (current_sums_mask >> s2) & 1 or (new_sums_mask >> s2) & 1:
                    valid = False
                else:
                    new_sums_mask |= (1 << s2)
            
            if valid:
                next_candidates = candidates[:i] + candidates[i+1:]
                if dfs(current_set + [c], current_sums_mask | new_sums_mask, next_candidates, target_k):
                    return True
                    
        return False

    current_target = best_size + 1
    
    while current_target <= 20:
        if time.time() > deadline - 5:
            break
            
        initial_sums_mask = 0
        for i, x in enumerate(prescribed):
            for y in prescribed[i:]:
                initial_sums_mask |= (1 << (x + y))
                
        rng_dfs = random.Random(current_target)
        shuffled_candidates = all_candidates[:]
        rng_dfs.shuffle(shuffled_candidates)
        
        found = dfs(list(prescribed), initial_sums_mask, shuffled_candidates, current_target)
        
        if found:
            current_target += 1
        else:
            break
            
    return best_set
