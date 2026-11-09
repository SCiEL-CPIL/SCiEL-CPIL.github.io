import random
import time

def construct():
    start_time = time.time()
    timeout = 140  # Safety margin for 150s limit
    
    seed_elements = [1, 2, 5, 11]
    best_set = seed_elements.copy()
    best_size = len(best_set)
    
    def get_sums(s):
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sums.add(s[i] + s[j])
        return sums
        
    candidates = [x for x in range(1, 285) if x not in seed_elements]
    
    while time.time() - start_time < timeout:
        current_set = seed_elements.copy()
        current_sums = get_sums(current_set)
        
        # Heuristic: Score candidates by number of conflicts with current_set
        scored_candidates = []
        for x in candidates:
            conflicts = 0
            for y in current_set:
                if (x + y) in current_sums:
                    conflicts += 1
            if (2 * x) in current_sums:
                conflicts += 1
            
            score = conflicts + random.random()
            scored_candidates.append((score, x))
        
        scored_candidates.sort()
        
        for _, x in scored_candidates:
            if x in current_set:
                continue
            
            new_sums = set()
            conflict = False
            for y in current_set:
                s = x + y
                if s in current_sums:
                    conflict = True
                    break
                new_sums.add(s)
            
            if not conflict:
                s_2x = 2 * x
                if s_2x in current_sums:
                    conflict = True
                else:
                    new_sums.add(s_2x)
            
            if not conflict:
                current_set.append(x)
                current_sums |= new_sums
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
                    if best_size == 20:
                        return best_set
        
    return best_set
