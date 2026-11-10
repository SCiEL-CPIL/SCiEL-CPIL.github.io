import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    n = 12
    m = 40
    
    # Start with a chain: {0}, {0,1}, {0,1,2}, ..., {0,...,11}
    chain = [list(range(k)) for k in range(n + 1)]
    
    current = list(chain)
    current_fs = [frozenset(s) for s in current]
    
    # All possible subsets
    all_subsets = []
    for k in range(n + 1):
        for combo in itertools.combinations(range(n), k):
            all_subsets.append(frozenset(combo))
    
    used = set(current_fs)
    
    # Greedily add sets that maximize comparability
    for _ in range(m - len(current)):
        if time.time() > deadline:
            break
        
        best = None
        best_score = -1
        for cand in all_subsets:
            if cand in used:
                continue
            # Count how many current sets are comparable with cand
            score = sum(1 for fs in current_fs if cand < fs or fs < cand)
            if score > best_score:
                best_score = score
                best = cand
        
        if best is None:
            break
        current.append(list(best))
        current_fs.append(best)
        used.add(best)
    
    # Ensure exactly m sets
    while len(current) < m:
        for cand in all_subsets:
            if cand not in used:
                current.append(list(cand))
                used.add(cand)
                break
        else:
            break
    
    return [list(s) for s in current]
