import random
import time

def construct():
    n = 12
    best_set = [1 << i for i in range(n)]
    best_max = max(best_set)
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    for _ in range(2000):
        if time.time() > deadline:
            break
            
        a = [1]
        valid = True
        
        for k in range(1, n):
            sums = {0}
            for x in a:
                new_sums = {s + x for s in sums}
                sums.update(new_sums)
            
            D = set()
            sums_list = list(sums)
            for i in range(len(sums_list)):
                for j in range(len(sums_list)):
                    D.add(abs(sums_list[i] - sums_list[j]))
            
            min_candidate = a[-1] + 1
            candidates = []
            max_search = best_max + 50 
            
            # Collect candidates
            while min_candidate <= max_search:
                if min_candidate not in D:
                    candidates.append(min_candidate)
                    if len(candidates) > 20:
                        break
                min_candidate += 1
                
            if not candidates:
                valid = False
                break
            
            # Strategy: pick smallest with high probability, 
            # but sometimes pick a larger one to explore space
            if random.random() < 0.7:
                a.append(candidates[0])
            else:
                a.append(random.choice(candidates))
                
        if valid:
            current_max = max(a)
            if current_max < best_max:
                best_max = current_max
                best_set = a[:]
                
    return best_set
