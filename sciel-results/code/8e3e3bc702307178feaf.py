import random
import time

def construct():
    deadline = time.time() + 55.0
    best_set = []
    
    # Try a few random orders to escape local minima
    for _ in range(10):
        if time.time() > deadline:
            break
            
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        S = []
        # sums[k] for current S
        # We don't strictly need to maintain sums incrementally if we recompute in is_valid
        # But maintaining helps if we do incremental checks.
        # Here we use full is_valid check for simplicity and correctness.
        
        for x in candidates:
            if time.time() > deadline:
                break
            S_new = S + [x]
            if _is_valid(S_new):
                S = S_new
                
        if len(S) > len(best_set):
            best_set = S
            
    return best_set

def _is_valid(S):
    n = len(S)
    if n < 3:
        return True
        
    # For each a in S, check if a is mean of subset of S\{a} of size >= 2
    for a in S:
        others = [y for y in S if y != a]
        k_max = len(others)
        
        # DP for subset sums of others
        # sums[k] is a bitset (int) where bit i is 1 if sum i is achievable with k elements
        sums = [0] * (k_max + 1)
        sums[0] = 1
        
        for y in others:
            # Update in reverse to avoid using y multiple times in same step
            for k in range(k_max, 0, -1):
                sums[k] |= (sums[k-1] << y)
                
        for k in range(2, k_max + 1):
            target = k * a
            if (sums[k] >> target) & 1:
                return False
                
    return True
