import random
import time

N = 256

def construct():
    start_time = time.time()
    time_limit = 58.0
    best_set = None
    best_size = N + 1
    
    while time.time() - start_time < time_limit:
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        current_set = []
        current_sums = set()
        
        for y in candidates:
            # Check 2*y
            if 2 * y in current_sums:
                continue
            
            # Check y + x
            blocked = False
            for x in current_set:
                if (y + x) in current_sums:
                    blocked = True
                    break
                    
            if blocked:
                continue
            
            # Add y
            current_set.append(y)
            current_sums.add(2 * y)
            for x in current_set[:-1]:
                current_sums.add(y + x)
                
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    if best_set is None:
        # Fallback to deterministic greedy
        selected = []
        sums = set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected
        
    return best_set
