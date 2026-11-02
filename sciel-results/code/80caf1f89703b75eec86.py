import itertools
import math
import time

def score_set(U):
    U = sorted(U)
    if 0 not in U:
        return -1
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    if len(sumset) == 0:
        return -1
    ratio = len(diffset) / len(sumset)
    max_u = max(U)
    if max_u == 0:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Exhaustive search for small sizes
    # Size 5, max 25
    for c in itertools.combinations(range(26), 5):
        if time.time() > deadline: break
        if 0 not in c: continue
        s = score_set(list(c))
        if s > best_score:
            best_score = s
            best_set = list(c)
            
    # Size 6, max 30
    for c in itertools.combinations(range(31), 6):
        if time.time() > deadline: break
        if 0 not in c: continue
        s = score_set(list(c))
        if s > best_score:
            best_score = s
            best_set = list(c)
            
    # Size 7, max 35
    # This is 3246321 combos. Might be tight.
    # Let's try.
    for c in itertools.combinations(range(36), 7):
        if time.time() > deadline: break
        if 0 not in c: continue
        s = score_set(list(c))
        if s > best_score:
            best_score = s
            best_set = list(c)
            
    # Hill climbing
    # Try to improve best_set by swapping one element
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = sorted(best_set)
        # Generate neighbors: remove one, add another
        for i in range(len(current)):
            for j in range(40): # max element 40
                if j in current: continue
                new_set = current[:i] + current[i+1:]
                new_set.append(j)
                new_set = sorted(set(new_set))
                if len(new_set) != len(current): continue
                if 0 not in new_set: continue
                s = score_set(new_set)
                if s > best_score + 1e-9:
                    best_score = s
                    best_set = new_set
                    improved = True
                    break
            if improved: break
            
    return best_set
