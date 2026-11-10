"""Combined program: bitmask 4-AP checking + effective search strategy."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
import numpy as np

N = 256
DEADLINE_FRACTION = 0.82  # safety margin

def _count_3ap(flags):
    """Count distinct 3-term APs (positive difference) using boolean array."""
    count = 0
    for a in range(1, N + 1):
        if not flags[a]:
            continue
        max_d = (N - a) // 2
        for d in range(1, max_d + 1):
            if flags[a + d] and flags[a + 2*d]:
                count += 1
    return count

def _check_add_4ap(flags, x):
    """Check if adding x to set (represented by flags) creates a 4-AP.
    Returns True if it would create a 4-AP."""
    # x is first: x, x+d, x+2d, x+3d
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if flags[x + d] and flags[x + 2*d] and flags[x + 3*d]:
            return True
    # x is second: x-d, x, x+d, x+2d
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        if flags[x - d] and flags[x + d] and flags[x + 2*d]:
            return True
    # x is third: x-2d, x-d, x, x+d
    max_d = min((x - 1) // 2, N - x)
    for d in range(1, max_d + 1):
        if flags[x - 2*d] and flags[x - d] and flags[x + d]:
            return True
    # x is fourth: x-3d, x-2d, x-d, x
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if flags[x - 3*d] and flags[x - 2*d] and flags[x - d]:
            return True
    return False

def _greedy_with_order(order, flags=None):
    """Greedy construction with a specific order."""
    if flags is None:
        flags = [False] * (N + 1)
    selected = []
    for x in order:
        if flags[x]:
            continue
        if not _check_add_4ap(flags, x):
            selected.append(x)
            flags[x] = True
    return selected

def _greedy_construct(order):
    """Greedy construction from scratch with given order."""
    flags = [False] * (N + 1)
    return _greedy_with_order(order, flags)

def _local_search(flags, current_score, deadline):
    """Try to improve by adding elements that don't create 4-AP and increase 3-AP count."""
    current_set = [i for i in range(1, N + 1) if flags[i]]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if flags[x]:
                continue
            if _check_add_4ap(flags, x):
                continue
            # Tentatively add x
            flags[x] = True
            new_score = _count_3ap(flags)
            if new_score > current_score:
                current_score = new_score
                improved = True
            else:
                flags[x] = False
    return current_score

def _swap_search(flags, current_score, deadline):
    """Try to improve by swapping elements: remove one, add another."""
    current_set = [i for i in range(1, N + 1) if flags[i]]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for elem in current_set:
            if time.time() > deadline:
                return current_score
            # Remove elem
            flags[elem] = False
            old_score = current_score
            
            best_swap = None
            best_new_score = current_score
            for x in range(1, N + 1):
                if flags[x]:
                    continue
                if _check_add_4ap(flags, x):
                    continue
                flags[x] = True
                new_score = _count_3ap(flags)
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_swap = x
                flags[x] = False
            
            if best_swap is not None and best_new_score > current_score:
                flags[best_swap] = True
                current_set.remove(elem)
                current_set.append(best_swap)
                current_score = best_new_score
                improved = True
            else:
                # Restore elem
                flags[elem] = True
    
    return current_score

def construct():
    start_time = time.time()
    deadline = start_time + 90 * DEADLINE_FRACTION  # ~73.8 seconds
    
    rng = np.random.default_rng(42)
    
    best_set = None
    best_score = -1
    
    # Strategy 1: Try various orderings for greedy
    orders_to_try = []
    
    # Natural order
    orders_to_try.append(list(range(1, N + 1)))
    # Reverse order
    orders_to_try.append(list(range(N, 0, -1)))
    
    # Random permutations
    for _ in range(30):
        orders_to_try.append(rng.permutation(N).tolist())
    
    # Structured: primes first
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [i for i in range(1, N + 1) if is_prime(i)]
    composites = [i for i in range(1, N + 1) if not is_prime(i)]
    orders_to_try.append(primes + composites)
    orders_to_try.append(composites + primes)
    
    # Powers of 2 and 3 first
    special = [2**k for k in range(1, 9) if 2**k <= N] + [3**k for k in range(1, 6) if 3**k <= N]
    rest = [i for i in range(1, N + 1) if i not in special]
    orders_to_try.append(special + rest)
    
    # Fibonacci-like
    fibs = [1, 2]
    while fibs[-1] + fibs[-2] <= N:
        fibs.append(fibs[-1] + fibs[-2])
    rest_fib = [i for i in range(1, N + 1) if i not in fibs]
    orders_to_try.append(fibs + rest_fib)
    
    for order in orders_to_try:
        if time.time() > deadline:
            break
        candidate = _greedy_construct(order)
        flags = [False] * (N + 1)
        for x in candidate:
            flags[x] = True
        score = _count_3ap(flags)
        if score > best_score:
            best_score = score
            best_set = candidate[:]
    
    # Strategy 2: Local search on best
    if best_set is not None and time.time() < deadline:
        flags = [False] * (N + 1)
        for x in best_set:
            flags[x] = True
        best_score = _local_search(flags, best_score, deadline)
        best_set = [i for i in range(1, N + 1) if flags[i]]
    
    # Strategy 3: More randomized greedy with local search
    for _ in range(50):
        if time.time() > deadline:
            break
        order = rng.permutation(N).tolist()
        candidate = _greedy_construct(order)
        flags = [False] * (N + 1)
        for x in candidate:
            flags[x] = True
        score = _count_3ap(flags)
        
        # Local search on this candidate
        score = _local_search(flags, score, deadline)
        if score > best_score:
            best_score = score
            best_set = [i for i in range(1, N + 1) if flags[i]]
    
    # Strategy 4: Swap search on best
    if best_set is not None and time.time() < deadline:
        flags = [False] * (N + 1)
        for x in best_set:
            flags[x] = True
        best_score = _swap_search(flags, best_score, deadline)
        best_set = [i for i in range(1, N + 1) if flags[i]]
    
    return best_set if best_set is not None else [1]

# EVOLVE-BLOCK-END
