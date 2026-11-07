import time
import random

_P0 = time.time()
_DEADLINE = _P0 + 85.0  # Safety margin before 90s kill

def _subset_products(nums):
    """Compute all 2^k subset products including empty product 1."""
    prods = {1}
    for x in nums:
        add = set()
        for p in prods:
            add.add(p * x)
        prods = prods | add
    return prods

def _can_add(prods, x):
    """Check if adding x keeps all subset products distinct."""
    for p in prods:
        if p * x in prods:
            return False
    return True

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _greedy(order, deadline):
    """Greedy: add numbers in the given order if they fit."""
    cur_set = []
    cur_prods = {1}
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(cur_prods, x):
            cur_set.append(x)
            new_prods = set()
            for p in cur_prods:
                new_prods.add(p * x)
            cur_prods = cur_prods | new_prods
    return cur_set, cur_prods

def _local_search(best_set, best_prods, deadline, max_iter=200):
    """Local search: try swaps to improve the set."""
    best_set = list(best_set)
    best_prods = set(best_prods)
    best_len = len(best_set)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        
        # Try single swaps: remove one element, try to add a different one
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            removed = best_set[i]
            remaining = best_set[:i] + best_set[i+1:]
            
            # Compute products of remaining
            rem_prods = _subset_products(remaining)
            
            # Try to add each candidate not in remaining
            for x in range(2, 41):
                if x in remaining:
                    continue
                if _can_add(rem_prods, x):
                    new_set = remaining + [x]
                    # Check if this improves (same size, but maybe enables future adds)
                    # For now, just accept if same size (it's a valid swap)
                    new_prods = _subset_products(new_set)
                    if len(new_set) >= best_len:
                        best_set = new_set
                        best_prods = new_prods
                        best_len = len(best_set)
                        improved = True
                        break
            if time.time() > deadline:
                break
        
        # Try to add any remaining element
        for x in range(2, 41):
            if x not in best_set:
                if _can_add(best_prods, x):
                    best_set.append(x)
                    new_prods = set()
                    for p in best_prods:
                        new_prods.add(p * x)
                    best_prods = best_prods | new_prods
                    best_len = len(best_set)
                    improved = True
        
        if not improved:
            break
    
    return best_set, best_prods

def construct():
    deadline = _DEADLINE
    
    # Generate multiple diverse orderings
    all_nums = list(range(2, 41))
    primes = [x for x in all_nums if _is_prime(x)]
    composites = [x for x in all_nums if not _is_prime(x)]
    
    orders = [
        # Basic orders
        all_nums[:],
        all_nums[::-1],
        # Primes first
        primes + composites,
        # Composites first
        composites + primes,
        # Alternating
        [],
    ]
    
    # Create alternating order
    pi, ci = 0, 0
    alt = []
    while pi < len(primes) or ci < len(composites):
        if pi < len(primes):
            alt.append(primes[pi])
            pi += 1
        if ci < len(composites):
            alt.append(composites[ci])
            ci += 1
    orders.append(alt)
    
    # Large primes first (harder to fit, so place them early)
    orders.append(sorted(all_nums, key=lambda x: -x))
    orders.append(sorted(all_nums, key=lambda x: x))
    
    # Randomized orders
    rng = random.Random(42)
    for _ in range(10):
        shuffled = all_nums[:]
        rng.shuffle(shuffled)
        orders.append(shuffled)
    
    # Run greedy on each ordering
    best_set = []
    best_prods = {1}
    
    for order in orders:
        if time.time() > deadline:
            break
        cur_set, cur_prods = _greedy(order, deadline)
        if len(cur_set) > len(best_set):
            best_set = cur_set
            best_prods = cur_prods
    
    # Local search from the best solution
    if time.time() < deadline - 5:
        best_set, best_prods = _local_search(best_set, best_prods, deadline)
    
    # Final attempt: try to add any remaining elements
    for x in range(2, 41):
        if x not in best_set and _can_add(best_prods, x):
            best_set.append(x)
            new_prods = set()
            for p in best_prods:
                new_prods.add(p * x)
            best_prods = best_prods | new_prods
    
    return list(best_set)
