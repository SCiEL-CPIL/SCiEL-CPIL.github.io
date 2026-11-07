import time
import random

def _valid_subset_sums(arr):
    """
    Returns True if no sum appears in two different cardinalities.
    Uses bitset DP for efficiency.
    """
    n = len(arr)
    if n == 0:
        return True
    # dp[k] is a bitset where bit s is 1 if sum s is achievable with exactly k elements
    dp = [0] * (n + 1)
    dp[0] = 1  # sum 0 with 0 elements
    
    for x in arr:
        # Iterate backwards to avoid using the new element multiple times in the same step
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
            
    # Check for collisions between different cardinalities
    seen = 0
    for k in range(n + 1):
        if dp[k] & seen:
            return False
        seen |= dp[k]
    return True

def _try_add(arr, x):
    """
    Returns True if adding x to arr maintains the cardinality-separation property.
    """
    # Optimization: If x is already in arr, it's not distinct
    if x in arr:
        return False
    # Quick check: The new set is arr + [x]
    # We can't easily update DP incrementally without storing it, 
    # so we recompute. For N < 40, this is fast.
    return _valid_subset_sums(arr + [x])

def _grow_greedy(arr, candidates, deadline):
    """
    Greedily add candidates from the list to arr if valid.
    Repeats passes until no more can be added or time runs out.
    """
    cur = list(arr)
    cur_set = set(cur)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in candidates:
            if time.time() > deadline:
                return cur
            if x in cur_set:
                continue
            if _try_add(cur, x):
                cur.append(x)
                cur_set.add(x)
                improved = True
    return cur

def _local_search_expand(arr, deadline):
    """
    Local search: Try to replace one element with two new elements (net +1 size).
    """
    cur = list(arr)
    n_max = 256
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(cur)
        # Try to remove cur[i] and add a, b
        for i in range(n):
            if time.time() > deadline:
                return cur
            x = cur[i]
            base = cur[:i] + cur[i+1:]
            base_set = set(base)
            
            # Try all pairs a, b
            # Optimization: Iterate a, then b > a
            for a in range(1, n_max + 1):
                if time.time() > deadline:
                    return cur
                if a == x or a in base_set:
                    continue
                if not _try_add(base, a):
                    continue
                base2 = base + [a]
                base2_set = base_set | {a}
                
                for b in range(a + 1, n_max + 1):
                    if time.time() > deadline:
                        return cur
                    if b == x or b in base2_set:
                        continue
                    if _try_add(base2, b):
                        cur = base2 + [b]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return cur

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    rng = random.Random(42)
    
    best = []
    
    # 1. Greedy Growth from various seeds and orders
    seeds = [
        [],
        [1, 2, 4, 8, 16, 32, 64, 128],
        [1, 3, 7, 15, 31, 63, 127, 255],
        [2, 5, 11, 23, 47, 95, 191],
    ]
    
    candidates_all = list(range(1, 257))
    orders = [
        candidates_all,
        list(reversed(candidates_all)),
    ]
    
    # Add a few random orders
    for _ in range(5):
        o = list(candidates_all)
        rng.shuffle(o)
        orders.append(o)
    
    for seed in seeds:
        if time.time() > deadline:
            break
        if not _valid_subset_sums(seed):
            continue
        for order in orders:
            if time.time() > deadline:
                break
            cur = _grow_greedy(seed, order, deadline)
            if len(cur) > len(best):
                best = list(cur)
                
    # 2. Local Search on the best found
    if time.time() < deadline - 5:
        # Run local search to expand
        best = _local_search_expand(best, deadline - 2)
        
        # After expansion, try to grow again with remaining candidates
        if time.time() < deadline - 2:
            best = _grow_greedy(best, candidates_all, deadline - 1)
            
        # Another round of local search
        if time.time() < deadline - 2:
            best = _local_search_expand(best, deadline - 1)
            
            # Final greedy pass
            if time.time() < deadline:
                best = _grow_greedy(best, candidates_all, deadline)

    # Ensure return type is list of Python ints
    return [int(v) for v in best]
