import time
import random
import math

def calc_score(U):
    if not U or 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0 # log(1/1)/log(1) is undefined, but limit is 1? 
        # Actually if U={0}, |U-U|=1, |U+U|=1, max=0. log(1)/log(1). 
        # Task says nonnegative integers. 
        # If max_u=0, denom log(1)=0. 
        # But U must contain 0. If U={0}, score is likely 1 or 0. 
        # Let's assume U has at least 2 elements or handle max_u=0 carefully.
        # If max_u > 0:
    diff_set = set()
    sum_set = set()
    for a in U:
        for b in U:
            diff_set.add(a - b)
            sum_set.add(a + b)
    
    num = len(diff_set)
    den = len(sum_set)
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1.0 + (math.log(num / den)) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 1, 2, 5],
        [0, 1, 2, 4, 5],
        list(range(10))
    ]
    
    for c in candidates:
        s = calc_score(c)
        if s > best_score:
            best_score = s
            best_set = list(c)
    
    # Local Search
    def local_search(current, max_iters=1000):
        cur_score = calc_score(current)
        cur_set = list(current)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random move: add, remove, or swap
            move_type = random.choice(['add', 'remove', 'swap'])
            
            new_set = list(cur_set)
            
            if move_type == 'add':
                # Try adding a small number
                max_val = max(new_set) if new_set else 0
                # Search range around current max
                candidates_to_add = [x for x in range(0, max_val + 10) if x not in new_set]
                if not candidates_to_add:
                    candidates_to_add = [max_val + 1]
                x = random.choice(candidates_to_add)
                new_set.append(x)
                
            elif move_type == 'remove':
                if len(new_set) > 1: # Keep 0
                    non_zero = [x for x in new_set if x != 0]
                    if non_zero:
                        x = random.choice(non_zero)
                        new_set.remove(x)
                    else:
                        continue
                        
            elif move_type == 'swap':
                if len(new_set) < 2:
                    continue
                non_zero = [x for x in new_set if x != 0]
                if not non_zero:
                    continue
                # Pick an element to replace
                idx = random.randint(0, len(non_zero)-1)
                old_val = non_zero[idx]
                # Pick a new value
                max_val = max(new_set)
                new_val = random.randint(1, max_val + 5)
                if new_val in new_set:
                    continue
                new_set[idx if old_val == new_set[0] else new_set.index(old_val)] = new_val
                # Ensure 0 is still there (it is, we didn't touch it unless 0 was in non_zero which is false)
                # Wait, new_set might have duplicates if logic is wrong. 
                # Let's use a set for construction then convert to list
                s_temp = set(new_set)
                s_temp.add(0)
                new_set = list(s_temp)
                if not new_set:
                    new_set = [0]

            new_score = calc_score(new_set)
            
            if new_score > cur_score:
                cur_score = new_score
                cur_set = new_set
                
        return cur_set, cur_score
    
    # Run multiple rounds of local search from different starts
    for i in range(20):
        if time.time() > deadline:
            break
        
        # Start from best or random
        if i % 2 == 0:
            start = list(best_set)
        else:
            # Random start
            size = random.randint(3, 10)
            start = [0]
            while len(start) < size:
                start.append(random.randint(1, 50))
            start = list(set(start))
            
        res_set, res_score = local_search(start, max_iters=500)
        
        if res_score > best_score:
            best_score = res_score
            best_set = res_set
            
        # Also try improving best_set directly
        if i % 5 == 0:
            res_set, res_score = local_search(best_set, max_iters=500)
            if res_score > best_score:
                best_score = res_score
                best_set = res_set

    return best_set
