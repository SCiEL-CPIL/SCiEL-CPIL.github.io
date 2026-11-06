import time
import random
from itertools import combinations

def _largest_sidon_subset(s):
    n = len(s)
    for k in range(n, 0, -1):
        for subset in combinations(range(n), k):
            sums = set()
            valid = True
            for i in range(k):
                si = s[subset[i]]
                for j in range(i, k):
                    sm = si + s[subset[j]]
                    if sm in sums:
                        valid = False
                        break
                    sums.add(sm)
                if not valid:
                    break
            if valid:
                return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    random.seed(42)
    
    best_set = None
    best_score = 12
    
    # Try various initial structures
    candidates = []
    # Dense APs
    for base in range(0, 120):
        for spread in range(1, 15):
            s = [base + i * spread for i in range(12)]
            if max(s) <= 128:
                candidates.append(s)
    # Random sets
    for _ in range(200):
        candidates.append(sorted(random.sample(range(0, 129), 12)))
    
    for s in candidates:
        if time.time() > deadline:
            break
        sc = _largest_sidon_subset(s)
        if sc < best_score:
            best_score = sc
            best_set = list(s)
    
    if best_set is None:
        best_set = sorted(random.sample(range(0, 129), 12))
        best_score = _largest_sidon_subset(best_set)
    
    current = list(best_set)
    current_score = best_score
    no_improve = 0
    
    while time.time() < deadline:
        idx = random.randint(0, 11)
        old_val = current[idx]
        new_val = random.randint(0, 128)
        
        if new_val == old_val or new_val in current:
            no_improve += 1
            continue
        
        current[idx] = new_val
        sc = _largest_sidon_subset(current)
        
        if sc < current_score:
            current_score = sc
            if sc < best_score:
                best_score = sc
                best_set = list(current)
            no_improve = 0
        else:
            current[idx] = old_val
            no_improve += 1
        
        if no_improve > 500:
            current = list(best_set)
            for _ in range(random.randint(1, 4)):
                idx2 = random.randint(0, 11)
                nv = random.randint(0, 128)
                if nv not in current:
                    current[idx2] = nv
            current_score = _largest_sidon_subset(current)
            no_improve = 0
    
    return best_set
