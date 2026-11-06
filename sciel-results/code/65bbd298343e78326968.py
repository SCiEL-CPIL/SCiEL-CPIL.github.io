import time
import random
from itertools import combinations

def _is_valid(selected):
    """Check if selected is a valid set."""
    n = len(selected)
    for idx_a, a in enumerate(selected):
        # Check if a divides sum of any 3 distinct others
        # Others are selected without a
        others = [v for i, v in enumerate(selected) if i != idx_a]
        if len(others) < 3:
            continue
        for b, c, d in combinations(others, 3):
            if (b + c + d) % a == 0:
                return False
    return True

def _can_add(x, S):
    """
    Check if x can be added to set S without violating the condition.
    S is a list of valid integers.
    """
    n = len(S)
    
    # 1. x does not divide sum of any 3 distinct elements in S
    if n >= 3:
        for i in range(n):
            si = S[i]
            for j in range(i+1, n):
                sj = S[j]
                for k in range(j+1, n):
                    sk = S[k]
                    if (si + sj + sk) % x == 0:
                        return False
    
    # 2. No a in S divides x + b + c for distinct b, c in S\{a}
    # Optimization: For each a, check if there exist b, c in S\{a}
    # such that (x + b + c) % a == 0
    # This is equivalent to (b + c) % a == (-x) % a
    for a in S:
        target = (-x) % a
        # Find all pairs (b, c) in S where b != a, c != a, b != c
        # and (b + c) % a == target
        others = [v for v in S if v != a]
        if len(others) < 2:
            continue
        
        # Use a set to track remainders we've seen
        seen_rems = set()
        for bi in others:
            bi_rem = bi % a
            needed = (target - bi_rem) % a
            if needed in seen_rems:
                return False
            seen_rems.add(bi_rem)
    
    return True

def _greedy(order, deadline):
    """Greedy construction following given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, pool, deadline, max_iters=100):
    """
    Improve selected by adding from pool or swapping.
    """
    selected = list(selected)
    selected_set = set(selected)
    best = selected
    best_len = len(selected)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        
        # Try to add
        for x in pool:
            if x in selected_set:
                continue
            if _can_add(x, selected):
                selected.append(x)
                selected_set.add(x)
                if len(selected) > best_len:
                    best = list(selected)
                    best_len = len(selected)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap
        # Remove y from selected, add x from pool
        # Only do this if it might lead to improvement? 
        # Actually, swap keeps size same. We only care about size increase.
        # But a swap might allow a subsequent add.
        # So we should try swaps that don't reduce size (they don't) 
        # but might change the set to allow more adds.
        
        # To avoid getting stuck in local minima of size, we can try swaps randomly.
        # But we need to be careful with time.
        
        # Let's try a few random swaps
        if len(selected) > 0 and len(pool) > 0:
            # Pick a random element to remove
            y = random.choice(selected)
            y_idx = selected.index(y)
            # Pick a random element to add
            x = random.choice(pool)
            
            # Check if x can be added to selected without y
            candidate = selected[:y_idx] + selected[y_idx+1:]
            if _can_add(x, candidate):
                candidate.append(x)
                # We don't know if this is "better" yet, but it's a new state.
                # We accept it if it's valid (it is, by construction)
                # We might want to bias towards swaps that involve larger numbers?
                # Let's just accept it and see if we can add more in next iteration.
                selected = candidate
                selected_set = set(selected)
                improved = True
                # Update best if size increased (it didn't, so no)
        
        if not improved:
            break
    
    return best

def _is_prime(x):
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0 or x % 3 == 0:
        return False
    i = 5
    while i * i <= x:
        if x % i == 0 or x % (i + 2) == 0:
            return False
        i += 6
    return True

def _num_divisors(x):
    cnt = 0
    for d in range(1, int(x**0.5) + 1):
        if x % d == 0:
            cnt += 1
            if d != x // d:
                cnt += 1
    return cnt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best = []
    best_len = 0
    
    def _update_best(s):
        nonlocal best, best_len
        if len(s) > best_len and _is_valid(s):
            best = list(s)
            best_len = len(s)
    
    # Phase 1: Greedy with different orderings
    all_nums = list(range(1, 193))
    
    # Order 1: Descending
    if time.time() < deadline:
        order = sorted(all_nums, reverse=True)
        res = _greedy(order, deadline)
        _update_best(res)
        
    # Order 2: Primes first (descending), then composites
    if time.time() < deadline:
        primes = [x for x in all_nums if _is_prime(x)]
        composites = [x for x in all_nums if not _is_prime(x)]
        order = sorted(primes, reverse=True) + sorted(composites, reverse=True)
        res = _greedy(order, deadline)
        _update_best(res)
        
    # Order 3: Few divisors first
    if time.time() < deadline:
        order = sorted(all_nums, key=lambda x: (_num_divisors(x), -x))
        res = _greedy(order, deadline)
        _update_best(res)
        
    # Phase 2: Local Search on best found so far
    if time.time() < deadline and best:
        pool = [x for x in all_nums if x not in set(best)]
        res = _local_search(best, pool, deadline, max_iters=200)
        _update_best(res)
        
    # Phase 3: Random Restart Greedy + Local Search
    rng = random.Random(42)
    while time.time() < deadline:
        # Random order, but biased towards larger numbers?
        # Just shuffle
        order = all_nums[:]
        rng.shuffle(order)
        
        # Maybe bias: sort by random weight + value
        # Let's try a mix
        if rng.random() < 0.5:
            order = sorted(order, key=lambda x: rng.random() + x/192.0, reverse=True)
        else:
            order = sorted(order, reverse=True) # Fallback to descending
            
        res = _greedy(order, deadline)
        if len(res) > best_len:
            _update_best(res)
            # Immediate local search
            if time.time() < deadline:
                pool = [x for x in all_nums if x not in set(res)]
                res_ls = _local_search(res, pool, deadline, max_iters=50)
                _update_best(res_ls)
                
    return best
