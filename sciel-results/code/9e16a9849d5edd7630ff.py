import random
import time

def construct():
    best_set = []
    best_size = 0
    start_time = time.time()
    time_limit = 110  # leave some buffer
    
    for seed in range(200):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(seed)
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        chosen = []
        counts = {}  # sum -> count of representations
        
        for x in candidates:
            # Check if adding x violates the constraint
            # New sums: x + y for y in chosen where y >= x (to avoid double counting)
            # Plus 2*x
            # Actually, we need to check all sums involving x:
            # For each y in chosen, the sum x+y gets one more representation
            # And 2*x gets one more representation (from x+x)
            
            # To be efficient, let's check if any sum would exceed 2
            ok = True
            new_sums = []
            
            for y in chosen:
                if y < x:
                    s = x + y
                    new_sums.append(s)
                elif y == x:
                    s = 2 * x
                    new_sums.append(s)
                else:
                    s = x + y
                    new_sums.append(s)
            
            # Also 2*x is already included if x is in chosen, but x is not in chosen yet
            # Wait, 2*x comes from the pair (x, x), which is a new representation
            new_sums.append(2 * x)
            
            # Check if any of these sums would have count > 2 after adding
            for s in new_sums:
                current_count = counts.get(s, 0)
                if current_count + 1 > 2:
                    ok = False
                    break
            
            if ok:
                chosen.append(x)
                for s in new_sums:
                    counts[s] = counts.get(s, 0) + 1
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen[:]
    
    return best_set
