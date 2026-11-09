import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    # Base solution: upper tail from 171 to 256
    base_set = set(range(171, 257))
    best_set = set(base_set)
    best_size = len(base_set)

    def can_add(x, S):
        """Check if x can be added to S without violating the property."""
        elems = sorted(S)
        
        # Case 1: x is 'a' (smallest)
        # x < b <= c
        for i, b in enumerate(elems):
            if b <= x:
                continue
            for c in elems[i:]: # c >= b
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is 'b' (middle)
        # a < x <= c
        for a in elems:
            if a >= x:
                break
            for c in elems:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is 'c' (largest)
        # a < b <= x
        for i, a in enumerate(elems):
            if a >= x:
                break
            for b in elems[i+1:]:
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
                    
        return True

    def get_divisor_count(x):
        count = 0
        i = 1
        while i * i <= x:
            if x % i == 0:
                count += 1
                if i != x // i:
                    count += 1
            i += 1
        return count

    candidates = list(range(1, 171))
    scored_candidates = [(get_divisor_count(x), x) for x in candidates]
    
    trials = 30
    for t in range(trials):
        if time.time() > deadline:
            break
            
        s_trial = set(base_set)
        
        if t == 0:
            order = sorted(scored_candidates, key=lambda x: x[0])
            cand_list = [x[1] for x in order]
        else:
            cand_list = candidates[:]
            random.shuffle(cand_list)
            
        for x in cand_list:
            if time.time() > deadline - 5:
                break
            if x in s_trial:
                continue
            if can_add(x, s_trial):
                s_trial.add(x)
                
        if len(s_trial) > best_size:
            best_size = len(s_trial)
            best_set = set(s_trial)

    # Final polish: try to add any remaining elements to the best set
    if best_size > len(base_set):
        s_final = set(best_set)
        for x in range(1, 257):
            if time.time() > deadline - 1:
                break
            if x not in s_final and can_add(x, s_final):
                s_final.add(x)
        if len(s_final) > best_size:
            best_set = s_final

    return list(best_set)
