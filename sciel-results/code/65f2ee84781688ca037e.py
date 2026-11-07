# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

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

def _compatible(n, S):
    """Check if n can be added to set S: for all distinct a,b in S, n does not divide a*b."""
    lst = list(S)
    L = len(lst)
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            if (lst[i] * lst[j]) % n == 0:
                return False
    return True

def _greedy_add(S, candidates, deadline):
    """Greedily add compatible candidates to S."""
    S = list(S)
    for n in candidates:
        if time.time() > deadline:
            break
        if n in S:
            continue
        if _compatible(n, S):
            S.append(n)
    return S

def construct():
    start = time.time()
    deadline = start + 50.0  # 50s budget, 10s safety margin
    
    N = 256
    all_nums = list(range(2, N + 1))
    primes = [x for x in all_nums if _is_prime(x)]
    composites = [x for x in all_nums if not _is_prime(x)]
    
    # Start with all primes
    S = set(primes)
    best = list(S)
    best_size = len(best)
    
    # Try greedy additions with different orderings
    orderings = [
        composites,
        list(reversed(composites)),
        sorted(composites, key=lambda x: sum(1 for d in range(2, x+1) if x % d == 0)),
        sorted(composites, key=lambda x: sum(1 for d in range(2, x+1) if x % d == 0), reverse=True),
        sorted(composites, key=lambda x: x),
        sorted(composites, key=lambda x: -x),
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        S_new = _greedy_add(set(primes), order, deadline)
        if len(S_new) > best_size:
            best_size = len(S_new)
            best = S_new
    
    # Local search: try removing one element and re-adding
    if time.time() < deadline:
        for _ in range(200):  # iterations
            if time.time() > deadline:
                break
            S_set = set(best)
            # Try removing each element and then greedily adding
            for x in list(S_set):
                if time.time() > deadline:
                    break
                S_reduced = S_set - {x}
                # Try adding various composites
                candidates = [c for c in composites if c not in S_reduced]
                random.shuffle(candidates)
                S_new = _greedy_add(S_reduced, candidates, deadline)
                if len(S_new) > best_size:
                    best_size = len(S_new)
                    best = S_new
                    S_set = set(best)
    
    # Also try randomized restarts
    if time.time() < deadline:
        for _ in range(50):
            if time.time() > deadline:
                break
            # Start with random subset of primes + some composites
            S_rand = set(random.sample(primes, max(1, len(primes) - 5)))
            for c in composites:
                if time.time() > deadline:
                    break
                if random.random() < 0.3:
                    if _compatible(c, S_rand):
                        S_rand.add(c)
            # Then greedily add more
            candidates = [x for x in all_nums if x not in S_rand]
            random.shuffle(candidates)
            S_new = _greedy_add(S_rand, candidates, deadline)
            if len(S_new) > best_size:
                best_size = len(S_new)
                best = S_new
    
    # Final validation
    result = list(set(best))
    # Verify validity
    L = len(result)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                if i == j or j == k or i == k:
                    continue
                if (result[j] * result[k]) % result[i] == 0:
                    # Invalid - fall back to primes
                    result = primes
                    break
            if result != best and len(result) < best_size:
                break
        if len(result) < best_size:
            break
    
    return [int(x) for x in sorted(result)]
# EVOLVE-BLOCK-END
