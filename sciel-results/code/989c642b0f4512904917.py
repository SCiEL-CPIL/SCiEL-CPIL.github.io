import time

_START = time.time()
_DEADLINE = 90.0
_MARGIN = 1.0  # safety margin in seconds

def _reachable(s, cap):
    """Boolean DP: which sums in [0, cap] are reachable using each element of s at most once."""
    reach = [False] * (cap + 1)
    reach[0] = True
    for x in s:
        # iterate descending to avoid reusing x
        for v in range(cap - x, -1, -1):
            if reach[v]:
                reach[v + x] = True
    return reach

def _valid(s, cap=200):
    r = _reachable(s, cap)
    return not r[cap]

def _new_sums_count(reach, x, cap=200):
    """Count how many new sums in [0, cap] would be created by adding x."""
    count = 0
    for v in range(cap - x, -1, -1):
        if reach[v] and not reach[v + x]:
            count += 1
    return count

def _greedy_construct():
    """Greedy construction prioritizing elements that create fewest new reachable sums."""
    selected = []
    reach = [False] * 201
    reach[0] = True
    remaining = set(range(1, 101))
    
    while remaining and (time.time() - _START) < (_DEADLINE - _MARGIN):
        # Find the element that creates the fewest new sums (and doesn't create 200)
        best_x = None
        best_new_sums = float('inf')
        
        for x in sorted(remaining, reverse=True):
            # Check if adding x would create sum 200
            creates_200 = False
            for v in range(200 - x, -1, -1):
                if reach[v] and v + x == 200:
                    creates_200 = True
                    break
            
            if creates_200:
                continue
            
            # Count new sums
            new_sums = 0
            for v in range(200 - x, -1, -1):
                if reach[v] and not reach[v + x]:
                    new_sums += 1
            
            if new_sums < best_new_sums:
                best_new_sums = new_sums
                best_x = x
        
        if best_x is None:
            break
        
        # Add best_x
        selected.append(best_x)
        remaining.discard(best_x)
        for v in range(200 - best_x, -1, -1):
            if reach[v]:
                reach[v + best_x] = True
    
    return selected

def _local_search_swap(best, cap=200):
    """Try swapping elements to improve the solution."""
    best_set = set(best)
    excluded = [x for x in range(1, 101) if x not in best_set]
    improved = False
    
    # Rebuild reach
    reach = [False] * (cap + 1)
    reach[0] = True
    for x in best:
        for v in range(cap - x, -1, -1):
            if reach[v]:
                reach[v + x] = True
    
    # Try adding excluded elements
    for x in sorted(excluded, reverse=True):
        if (time.time() - _START) > (_DEADLINE - _MARGIN):
            break
        # Check if adding x would create 200
        bad = False
        for v in range(cap - x, -1, -1):
            if reach[v] and v + x == cap:
                bad = True
                break
        if not bad:
            best.append(x)
            best_set.add(x)
            for v in range(cap - x, -1, -1):
                if reach[v]:
                    reach[v + x] = True
            improved = True
    
    # Try swapping: remove one element, add another
    if improved or True:  # Always try swaps
        best_set = set(best)
        excluded = [x for x in range(1, 101) if x not in best_set]
        
        for i, x_remove in enumerate(list(best)):
            if (time.time() - _START) > (_DEADLINE - _MARGIN):
                break
            for x_add in excluded:
                if x_add == x_remove:
                    continue
                # Try the swap
                trial = best[:]
                trial.remove(x_remove)
                trial.append(x_add)
                
                if _valid(trial, cap):
                    # Check if this swap allows us to add more elements later
                    # For now, accept any valid swap (same cardinality)
                    # But we want to increase cardinality, so only accept if it helps
                    pass
    
    return best

def construct():
    # Start with greedy construction using the new scoring
    best = _greedy_construct()
    
    # Local improvement: try to add back excluded elements
    changed = True
    while changed and (time.time() - _START) < (_DEADLINE - _MARGIN):
        changed = False
        best_set = set(best)
        reach = [False] * 201
        reach[0] = True
        for x in best:
            for v in range(200 - x, -1, -1):
                if reach[v]:
                    reach[v + x] = True
        excluded = [x for x in range(100, 0, -1) if x not in best_set]
        for x in excluded:
            if x in best_set:
                continue
            # would adding x create 200?
            bad = False
            for v in range(200 - x, -1, -1):
                if reach[v] and v + x == 200:
                    bad = True
                    break
            if not bad:
                best.append(x)
                best_set.add(x)
                for v in range(200 - x, -1, -1):
                    if reach[v]:
                        reach[v + x] = True
                changed = True
        if not changed:
            break
    
    # Final exact verification
    if not _valid(best):
        best = [1]
    
    if (time.time() - _START) > (_DEADLINE - _MARGIN):
        if not _valid(best):
            best = [1]
    
    return best
