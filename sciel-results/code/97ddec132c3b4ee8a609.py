import time

def is_valid_fast(s):
    """Fast cross-cardinality subset sum check using DP."""
    k = len(s)
    sums_by_card = [set() for _ in range(k + 1)]
    sums_by_card[0].add(0)
    
    for x in s:
        for c in range(k, 0, -1):
            for sm in list(sums_by_card[c-1]):
                sums_by_card[c].add(sm + x)
    
    # Check cross-cardinality disjointness
    seen = set()
    for c in range(k + 1):
        for sm in sums_by_card[c]:
            if sm in seen:
                return False
            seen.add(sm)
    return True

def construct():
    start = time.time()
    deadline = start + 85
    
    # Start with powers of 2 (guaranteed valid, all subset sums unique)
    base = [1, 2, 4, 8, 16, 32, 64, 128]
    best = base[:]
    
    # Greedy extension
    used = set(base)
    for x in range(1, 257):
        if time.time() > deadline - 5:
            break
        if x in used:
            continue
        new_set = base + [x]
        if is_valid_fast(new_set):
            base = new_set
            used.add(x)
    
    best = base[:]
    
    # Try to find additional elements via different starting points
    # and local search
    
    # Phase 1: Try all possible single additions to current best
    used = set(best)
    for x in range(1, 257):
        if time.time() > deadline - 5:
            break
        if x in used:
            continue
        new_set = best + [x]
        if is_valid_fast(new_set):
            best = new_set
            used.add(x)
    
    # Phase 2: Local search with swaps to enable growth
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        
        n = len(best)
        improved = False
        
        # Try to extend
        used = set(best)
        for x in range(1, 257):
            if time.time() > deadline - 10:
                break
            if x in used:
                continue
            new_set = best + [x]
            if is_valid_fast(new_set):
                best = new_set
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove element i, add element x
        for i in range(n):
            if time.time() > deadline - 10:
                break
            for x in range(1, 257):
                if time.time() > deadline - 10:
                    break
                if x == best[i] or x in best:
                    continue
                new_set = [best[j] for j in range(n) if j != i] + [x]
                if is_valid_fast(new_set):
                    # Try to extend after swap
                    used = set(new_set)
                    for z in range(1, 257):
                        if z in used:
                            continue
                        ext_set = new_set + [z]
                        if is_valid_fast(ext_set):
                            best = ext_set
                            improved = True
                            break
                    if improved:
                        break
        if improved:
            continue
        
        # If no improvement after many iterations, try different strategy
        if iteration > 50:
            break
    
    return best
