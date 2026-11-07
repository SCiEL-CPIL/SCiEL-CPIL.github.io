import time
import random

def _init_dp(arr):
    n = len(arr)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    return dp

def _is_valid_dp(dp):
    acc = 0
    for k in range(len(dp)):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True

def _verify_add(dp, n, x):
    """
    Checks if adding x to a set of size n (represented by dp) is valid.
    Returns new_dp if valid, None otherwise.
    """
    new_dp = [0] * (n + 2)
    for k in range(n + 1):
        new_dp[k] = dp[k]
    
    for k in range(n + 1, 0, -1):
        new_dp[k] |= (dp[k - 1] << x)
        
    acc = 0
    for k in range(n + 2):
        if new_dp[k] & acc:
            return None
        acc |= new_dp[k]
        
    return new_dp

def _grow_greedy(start, strategy='smallest', deadline=None, rng=None):
    cur = list(start)
    cur_set = set(cur)
    n = len(cur)
    dp = _init_dp(cur)
    
    improved = True
    while improved:
        if deadline is not None and time.monotonic() > deadline:
            break
        improved = False
        
        candidates = [v for v in range(1, 257) if v not in cur_set]
        if not candidates:
            break
            
        if strategy == 'smallest':
            order = candidates
        elif strategy == 'largest':
            order = list(reversed(candidates))
        elif strategy == 'random':
            order = candidates[:]
            if rng:
                rng.shuffle(order)
            else:
                random.shuffle(order)
        elif strategy == 'scored':
            # Score candidates by how "safe" they are (fewer conflicts)
            scored = []
            for v in candidates:
                new_dp = _verify_add(dp, n, v)
                if new_dp is not None:
                    # Count total bits set in new_dp as a "complexity" measure
                    # Lower complexity = more room for future additions
                    total_bits = sum(bin(x).count('1') for x in new_dp)
                    scored.append((total_bits, v))
            scored.sort()
            order = [v for _, v in scored]
        else:
            order = candidates
            
        for v in order:
            new_dp = _verify_add(dp, n, v)
            if new_dp is not None:
                cur.append(v)
                cur_set.add(v)
                n += 1
                dp = new_dp
                improved = True
                break
                
    return cur

def _local_search(cur, deadline):
    """
    Local search: try 1-for-1 swaps to escape local optima.
    More efficient than _polish by caching DP computations.
    """
    cur = list(cur)
    n = len(cur)
    dp = _init_dp(cur)
    cur_set = set(cur)
    
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        
        # Precompute DP for each element removed (cache)
        rem_dps = []
        for i in range(n):
            remaining = cur[:i] + cur[i+1:]
            rem_dp = _init_dp(remaining)
            rem_dps.append(rem_dp)
        
        # Try replacing each element with a different value
        for i in range(n):
            if time.monotonic() > deadline:
                break
            old = cur[i]
            rem_dp = rem_dps[i]
            remaining_set = set(cur[:i] + cur[i+1:])
            
            # Try all values not in remaining set
            for v in range(1, 257):
                if v in remaining_set:
                    continue
                
                new_dp = _verify_add(rem_dp, n - 1, v)
                if new_dp is not None:
                    # Found a valid swap - check if it helps us grow further
                    cur_new = cur[:i] + [v] + cur[i+1:]
                    cur_new.sort()
                    
                    # Try to grow from this new state
                    grown = _grow_greedy(cur_new, 'smallest', deadline)
                    if len(grown) > n:
                        cur = grown
                        n = len(cur)
                        dp = _init_dp(cur)
                        cur_set = set(cur)
                        improved = True
                        break
            if improved:
                break
    return cur

def _local_search_multi(cur, deadline, rng):
    """
    Multi-strategy local search with random restarts within the neighborhood.
    """
    cur = list(cur)
    n = len(cur)
    
    # Try multiple random perturbations
    for attempt in range(30):
        if time.monotonic() > deadline:
            break
        
        # Random perturbation: swap 1-2 random elements
        perturbed = cur[:]
        num_swaps = rng.randint(1, 2)
        for _ in range(num_swaps):
            i = rng.randint(0, len(perturbed) - 1)
            old = perturbed[i]
            remaining_set = set(perturbed) - {old}
            # Find a valid replacement
            found = False
            for _ in range(50):  # Try random replacements
                v = rng.randint(1, 256)
                if v in remaining_set:
                    continue
                test = perturbed[:i] + [v] + perturbed[i+1:]
                test.sort()
                test_dp = _init_dp(test)
                if _is_valid_dp(test_dp):
                    perturbed[i] = v
                    found = True
                    break
            if not found:
                perturbed[i] = old  # Revert
        
        # Verify perturbation is valid
        perturbed.sort()
        perturbed_dp = _init_dp(perturbed)
        if not _is_valid_dp(perturbed_dp):
            continue
        
        # Try to grow from perturbation
        grown = _grow_greedy(perturbed, 'smallest', deadline)
        if len(grown) > len(cur):
            cur = grown
            n = len(cur)
    
    return cur

def construct():
    t0 = time.monotonic()
    deadline = t0 + 85.0  # Safety margin before 90s kill
    best = None
    
    def update_best(arr):
        nonlocal best
        if best is None or len(arr) > len(best):
            best = list(arr)
    
    # 1. Deterministic Greedies with multiple strategies
    for strategy in ['smallest', 'largest', 'scored']:
        if time.monotonic() > deadline:
            break
        res = _grow_greedy([], strategy, deadline)
        update_best(res)
    
    # Start from Powers of 2
    p2 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for strategy in ['smallest', 'largest', 'scored']:
        if time.monotonic() > deadline:
            break
        res = _grow_greedy(p2, strategy, deadline)
        update_best(res)
    
    # Start from small consecutive sets
    for start_size in [1, 2, 3, 4]:
        for start in range(1, min(257 - start_size + 1, 50)):
            if time.monotonic() > deadline:
                break
            initial = list(range(start, start + start_size))
            res = _grow_greedy(initial, 'smallest', deadline)
            update_best(res)
    
    # 2. Randomized Restarts with Greedy Growth
    rng = random.Random(42)
    if time.monotonic() < deadline:
        for attempt in range(300):
            if time.monotonic() > deadline - 2:
                break
            k = rng.randint(0, 6)
            if k == 0:
                start = []
            else:
                start = rng.sample(range(1, 257), k)
                start.sort()
                start_dp = _init_dp(start)
                if not _is_valid_dp(start_dp):
                    continue
            
            strategy = rng.choice(['smallest', 'largest', 'scored', 'random'])
            res = _grow_greedy(start, strategy, deadline, rng)
            update_best(res)
    
    # 3. Local search on best solution
    if best is not None and time.monotonic() < deadline - 10:
        polished = _local_search(best, deadline)
        update_best(polished)
    
    # 4. Multi-strategy local search with random perturbations
    if best is not None and time.monotonic() < deadline - 5:
        multi = _local_search_multi(best, deadline, rng)
        update_best(multi)
    
    # 5. Final greedy growth on the polished result
    if best is not None and time.monotonic() < deadline - 1:
        final = _grow_greedy(best, 'smallest', deadline)
        update_best(final)
    
    if best is None:
        best = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    return best
