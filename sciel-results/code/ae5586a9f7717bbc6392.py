import time
import random

def construct():
    deadline = time.time() + 140
    best_set = []
    best_size = 0
    N = 1000
    max_sum = 2 * N
    
    while time.time() < deadline:
        counts = [0] * (max_sum + 1)
        chosen = []
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            ok = True
            # Check sums x + y
            for y in chosen:
                s = x + y
                if counts[s] >= 2:
                    ok = False
                    break
            if ok:
                s = 2 * x
                if counts[s] >= 2:
                    ok = False
            
            if ok:
                # Update counts
                for y in chosen:
                    counts[x + y] += 1
                counts[2 * x] += 1
                chosen.append(x)
                
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen[:]
            
    return best_set
