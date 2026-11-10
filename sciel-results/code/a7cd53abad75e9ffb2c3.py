import random
import time

def has_zero_sum_subset(diffs):
    """
    Check if any subset of diffs sums to 0.
    Since all diffs are non-zero, any such subset has size >= 2.
    """
    if not diffs:
        return False
    reachable = {0}
    for d in diffs:
        new_reachable = set()
        for s in reachable:
            new_reachable.add(s)
            new_reachable.add(s + d)
        reachable = new_reachable
        if 0 in reachable:
            # We found a subset summing to 0.
            # We need to ensure the subset size is >= 2.
            # Since d != 0, a single element cannot sum to 0.
            # So any subset summing to 0 has size >= 2.
            return True
    return False

def is_valid(S):
    """
    Check if no element in S is the mean of a subset of at least 2 other elements.
    """
    n = len(S)
    for i in range(n):
        y = S[i]
        # Construct list of differences to other elements
        diffs = []
        for j in range(n):
            if i == j:
                continue
            diffs.append(S[j] - y)
        if has_zero_sum_subset(diffs):
            return False
    return True

def construct():
    start_time = time.time()
    best_set = []
    
    # We try to find a set of size up to 48.
    # Use randomized greedy with restarts.
    
    max_restarts = 2000
    for _ in range(max_restarts):
        if time.time() - start_time > 50:
            break
            
        S = []
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(S) >= 48:
                break
            # Check if adding x keeps the set valid
            # We only need to check if x is valid in S+[x]
            # AND if existing elements remain valid.
            # is_valid(S + [x]) checks all.
            if is_valid(S + [x]):
                S.append(x)
        
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == 48:
                break
                
    return best_set
