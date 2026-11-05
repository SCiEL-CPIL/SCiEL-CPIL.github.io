import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Initial greedy pass (deterministic)
    chosen, counts = [], {}
    for x in range(1, 1001):
        valid = True
        for y in chosen:
            s = x + y
            if counts.get(s, 0) + 1 > 2:
                valid = False
                break
        if valid:
            s = 2 * x
            if counts.get(s, 0) + 1 > 2:
                valid = False
        if valid:
            for y in chosen:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            s = 2 * x
            counts[s] = counts.get(s, 0) + 1
            chosen.append(x)
    
    best_set = chosen
    best_size = len(chosen)
    
    # Randomized greedy with restarts
    for _ in range(50):  # Limit restarts to fit in time
        if time.time() > deadline:
            break
            
        order = random.sample(range(1, 1001), 1000)
        chosen, counts = [], {}
        for x in order:
            valid = True
            for y in chosen:
                s = x + y
                if counts.get(s, 0) + 1 > 2:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if counts.get(s, 0) + 1 > 2:
                    valid = False
            if valid:
                for y in chosen:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                s = 2 * x
                counts[s] = counts.get(s, 0) + 1
                chosen.append(x)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen
    
    return best_set
