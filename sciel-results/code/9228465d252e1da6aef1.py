"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

# Precompute prime exponent vectors for numbers 2..40
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_NUM_PRIMES = len(_PRIMES)

def _factor_vector(n):
    """Return exponent vector of n over the primes list."""
    vec = [0] * _NUM_PRIMES
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return tuple(vec)

# Precompute vectors for all candidates 2..40
_CANDIDATES = list(range(2, 41))
_VECTORS = {}
for _c in _CANDIDATES:
    _VECTORS[_c] = _factor_vector(_c)

def _check_distinct_subset_products(vecs):
    """Check if all 2^len(vecs) subset sums of exponent vectors are distinct."""
    k = len(vecs)
    if k == 0:
        return True
    # Use set of tuple sums
    sums = set()
    # Incremental: start with {zero}, add each vector
    current = {tuple([0] * _NUM_PRIMES)}
    for i in range(k):
        v = vecs[i]
        new_sums = set()
        for s in current:
            ns = tuple(s[j] + v[j] for j in range(_NUM_PRIMES))
            new_sums.add(ns)
        # Check for collision: any new sum already in current or in new_sums
        for ns in new_sums:
            if ns in current:
                return False
        current.update(new_sums)
        if len(current) != 2 ** (i + 1):
            return False
    return True

def _try_greedy(candidates, vecs_map, order):
    """Greedy: add candidates in given order if they preserve distinctness."""
    selected = []
    selected_vecs = []
    for c in order:
        if c not in candidates:
            continue
        new_vecs = selected_vecs + [vecs_map[c]]
        if _check_distinct_subset_products(new_vecs):
            selected.append(c)
            selected_vecs.append(vecs_map[c])
    return selected

def _random_order(candidates):
    """Shuffle candidates with some bias toward primes and small numbers."""
    lst = list(candidates)
    random.shuffle(lst)
    return lst

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    best_result = []
    best_len = 0

    # Precompute all candidate vectors
    vecs_map = _VECTORS

    # Start with a deterministic greedy: try primes first, then composites
    primes_list = [p for p in _PRIMES if p <= 40]
    composites_list = [c for c in _CANDIDATES if c not in _PRIMES]

    # Deterministic greedy 1: primes in order, then composites in order
    order1 = primes_list + composites_list
    res1 = _try_greedy(set(_CANDIDATES), vecs_map, order1)
    if len(res1) > best_len:
        best_len = len(res1)
        best_result = res1

    # Deterministic greedy 2: composites first (sorted by number of prime factors)
    def num_prime_factors(n):
        cnt = 0
        for p in _PRIMES:
            while n % p == 0:
                cnt += 1
                n //= p
        return cnt

    composites_sorted = sorted(composites_list, key=lambda c: (num_prime_factors(c), c))
    order2 = primes_list + composites_sorted
    res2 = _try_greedy(set(_CANDIDATES), vecs_map, order2)
    if len(res2) > best_len:
        best_len = len(res2)
        best_result = res2

    # Randomized restarts
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        order = _random_order(_CANDIDATES)
        res = _try_greedy(set(_CANDIDATES), vecs_map, order)
        if len(res) > best_len:
            best_len = len(res)
            best_result = res
            # Try to extend: if we found a good set, try adding more
            # by doing another greedy pass with remaining candidates
            remaining = [c for c in _CANDIDATES if c not in best_result]
            for c in remaining:
                if time.time() > deadline:
                    break
                test_vecs = [vecs_map[x] for x in best_result] + [vecs_map[c]]
                if _check_distinct_subset_products(test_vecs):
                    best_result.append(c)
                    best_len += 1

    # Final safety: ensure we return at least the primes
    if not best_result:
        best_result = primes_list[:]

    # Verify final result
    final_vecs = [vecs_map[x] for x in best_result]
    if not _check_distinct_subset_products(final_vecs):
        # Fallback to primes only
        best_result = primes_list[:]

    return list(best_result)

# EVOLVE-BLOCK-END
