import time
import random
import math

def score(U):
    if not U:
        return 0.0
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0 # log(1)/log(1) is undefined, but limit is 1? 
                   # Actually if U={0}, U+U={0}, U-U={0}. Ratio 1. Log 0. Score 1.
    
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
            
    size_sum = len(sumset)
    size_diff = len(diffset)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    if ratio <= 1:
        return 1.0 # Log is <= 0, score <= 1. We want > 1.
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    val = 1.0 + math.log(ratio) / denom
    return val

def local_search(U, max_iter=5000):
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(set(U))
    
    current_score = score(U)
    best_score = current_score
    best_U = list(U)
    
    max_val = max(U) * 2 if max(U) > 0 else 10
    
    for _ in range(max_iter):
        # Pick a random element to change (keep 0 fixed)
        if len(U) > 1:
            idx = random.randint(1, len(U) - 1)
            old_val = U[idx]
            # Try a random new value
            # Sometimes step small, sometimes big
            if random.random() < 0.5:
                new_val = max(0, old_val + random.randint(-max(1, old_val//10), max(1, old_val//10)))
            else:
                new_val = random.randint(0, max_val)
            
            new_U = U.copy()
            new_U[idx] = new_val
            new_U = sorted(set(new_U))
            
            if len(new_U) != len(U):
                # Size changed, might be ok but we prefer fixed size for comparison
                # Or we can allow it. The score depends on size implicitly.
                # Let's allow it.
                pass
            
            s = score(new_U)
            if s > best_score:
                best_score = s
                best_U = new_U
                current_score = s
                U = new_U
                # Expand search range if we found something good
                if max(U) > max_val:
                    max_val = max(U) * 2

    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    best_U = [0, 1, 4, 16]
    best_score = score(best_U)
    
    # Known good seeds
    seeds = [
        [0, 1, 3],
        [0, 1, 4, 16],
        [0, 1, 3, 9, 27],
        [0, 1, 5, 15],
        [0, 1, 6, 18],
        [0, 1, 10],
        [0, 1, 2, 7, 18], # Random
    ]
    
    for seed in seeds:
        s = score(seed)
        if s > best_score:
            best_score = s
            best_U = list(seed)
            
    # Local search on best seed
    if time.time() < deadline - 10:
        U, s = local_search(best_U, max_iter=2000)
        if s > best_score:
            best_score = s
            best_U = U

    # Try different sizes with random restarts
    for n in [3, 4, 5, 6]:
        if time.time() > deadline - 15:
            break
        for restart in range(5):
            if time.time() > deadline - 15:
                break
            # Generate random set of size n
            # Start with geometric-like
            base = random.randint(2, 5)
            U = [0]
            cur = 1
            for _ in range(n - 1):
                U.append(cur)
                cur *= base
                # Add some noise
                if random.random() < 0.3:
                    U.append(random.randint(0, max(U) * 2))
                    U = sorted(set(U))
                    if len(U) > n:
                        U.pop() # Keep size roughly n
            
            U, s = local_search(U, max_iter=1000)
            if s > best_score:
                best_score = s
                best_U = U
                
    return best_U
