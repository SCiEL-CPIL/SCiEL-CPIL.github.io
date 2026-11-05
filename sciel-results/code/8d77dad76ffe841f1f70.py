import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    n = 4096
    best_set = []
    best_size = 0
    
    # Helper to check if x can be added
    def can_add(x, S, existing_sums):
        # New sums involving x
        # (x, x, x)
        s_1 = 3 * x
        if s_1 in existing_sums:
            return False, None
        
        new_sums = {s_1}
        
        # (x, x, a) for a in S
        for a in S:
            s_2 = 2 * x + a
            if s_2 in existing_sums or s_2 in new_sums:
                return False, None
            new_sums.add(s_2)
            
        # (x, a, b) for a, b in S, a <= b
        k = len(S)
        for i in range(k):
            a = S[i]
            for j in range(i, k):
                b = S[j]
                s_3 = x + a + b
                if s_3 in existing_sums or s_3 in new_sums:
                    return False, None
                new_sums.add(s_3)
                
        return True, new_sums

    while time.time() < deadline:
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        current_set = []
        current_sums = set()
        
        for x in candidates:
            if len(current_set) + (len(candidates) - candidates.index(x)) <= best_size:
                # Optional pruning: if we can't beat best, break?
                # Hard to calculate index efficiently in loop. Skip.
                pass
                
            ok, new_sums = can_add(x, current_set, current_sums)
            if ok:
                current_set.append(x)
                current_sums.update(new_sums)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
            # If we reached a theoretical high, we might stop early?
            # But let's keep searching to ensure we found the max.
            
    return best_set
