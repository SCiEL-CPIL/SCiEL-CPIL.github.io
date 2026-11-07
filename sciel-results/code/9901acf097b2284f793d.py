import time
import random

def _valid(S):
    n = len(S)
    if n == 0:
        return True
    if len(set(S)) != n:
        return False
    if any(x < 1 or x > 256 for x in S):
        return False
    
    # Generate all subset sums
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += S[i]
        sums.add(s)
    
    slist = sorted(sums)
    m = len(slist)
    
    # Check no two distinct sums divide each other
    for i in range(m):
        a = slist[i]
        # For each multiple of a, check if it's in slist
        # But simpler: just check all j > i
        for j in range(i + 1, m):
            if slist[j] % a == 0:
                return False
    return True

def construct():
    start = time.time()
    deadline = start + 85.0
    
    best = [11, 13, 17]
    best_size = 3
    
    random.seed(123)
    
    # Precompute valid small sets to start from
    # Try greedy with different random orderings
    for trial in range(300):
        if time.time() > deadline - 5.0:
            break
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        current = []
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if len(current) >= 12:
                break
            trial_set = current + [x]
            if _valid(trial_set):
                current = trial_set
        
        if len(current) > best_size:
            best = list(current)
            best_size = len(current)
    
    # Try to extend best set
    if time.time() < deadline - 3.0:
        cur = list(best)
        for x in range(1, 257):
            if time.time() > deadline - 3.0:
                break
            if x in cur:
                continue
            if len(cur) >= 15:
                break
            trial = cur + [x]
            if _valid(trial):
                cur = trial
                best = list(cur)
                best_size = len(cur)
    
    if _valid(best):
        return [int(x) for x in best]
    return [int(x) for x in [11, 13, 17]]
