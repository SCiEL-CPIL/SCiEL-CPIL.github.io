import numpy as np
import random
import time

def score_set(U):
    U = list(set(U))
    if 0 not in U:
        return -1
    U.sort()
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    
    sum_size = len(sumset)
    diff_size = len(diffset)
    
    if sum_size == 0:
        return -1
        
    ratio = diff_size / sum_size
    if ratio <= 0:
        return -1
        
    log_ratio = np.log2(ratio)
    log_max = np.log2(2 * max_u + 1)
    
    if log_max == 0:
        return -1
        
    return 1 + log_ratio / log_max

def construct():
    # Start with a small set and grow/optimize using Hill Climbing
    current_set = [0, 1, 3, 7]
    current_score = score_set(current_set)
    best_set = current_set[:]
    best_score = current_score
    
    max_time = 110
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        improved = False
        
        # 1. Try removing elements
        for i in range(1, len(current_set)):
            new_set = current_set[:i] + current_set[i+1:]
            s = score_set(new_set)
            if s > current_score:
                current_set = new_set
                current_score = s
                improved = True
                break
        
        if improved:
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
            continue
        
        # 2. Try adding elements
        max_u = max(current_set)
        candidates = set()
        for x in current_set:
            for delta in [-2, -1, 1, 2, 3]:
                val = x + delta
                if val > 0:
                    candidates.add(val)
        for _ in range(5):
            candidates.add(random.randint(1, 2 * max_u + 10))
        candidates.add(max_u + 1)
        candidates.add(max_u * 2)
        candidates.add(max_u * 2 + 1)
        
        for c in candidates:
            if c in current_set:
                continue
            new_set = current_set + [c]
            s = score_set(new_set)
            if s > current_score:
                current_set = new_set
                current_score = s
                improved = True
                break
        
        if improved:
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
            continue
        
        # 3. Try swapping elements
        if len(current_set) > 2:
            idx = random.randint(1, len(current_set) - 1)
            old_val = current_set[idx]
            for _ in range(10):
                new_val = random.randint(1, 2 * max_u)
                if new_val == old_val or new_val in current_set:
                    continue
                new_set = current_set[:]
                new_set[idx] = new_val
                new_set.sort()
                s = score_set(new_set)
                if s > current_score:
                    current_set = new_set
                    current_score = s
                    improved = True
                    break
            
            if improved:
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
                continue
        
        # 4. Random kick to escape local minima
        if len(current_set) > 2:
            idx = random.randint(1, len(current_set) - 1)
            new_set = current_set[:idx] + current_set[idx+1:]
            new_val = random.randint(1, 3 * max_u)
            if new_val not in new_set:
                new_set.append(new_val)
                s = score_set(new_set)
                if s > current_score:
                    current_set = new_set
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current_set[:]
                    continue

    return best_set
