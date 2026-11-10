"""Combined program: Prime base with multi-strategy greedy composite expansion."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True

def _prime_factors(n):
    """Return sorted list of prime factors of n (with multiplicity)."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def _has_distinct_subset_products(S):
    """Check that all 2^k subset products of S are distinct."""
    if len(S) == 0:
        return True
    prods = [1]
    for a in S:
        new_prods = [x * a for x in prods]
        prods.extend(new_prods)
    return len(prods) == len(set(prods))

def _fast_check(S, candidate):
    """
    Quick check: if candidate's prime factors are all present as elements in S,
    then candidate is a product of elements in S, so subset products won't be distinct.
    This is a necessary (not sufficient) condition for validity.
    """
    cand_factors = set(_prime_factors(candidate))
    s_set = set(S)
    # If all prime factors of candidate are in S, candidate = product of some subset of S
    if cand_factors.issubset(s_set):
        return False
    return True

def _greedy_build(primes, composites, order='asc', max_size=18):
    """
    Greedily build a set starting from primes, adding composites in given order.
    order: 'asc', 'desc', 'fewest_factors', 'most_factors'
    """
    S = list(primes)
    
    if not _has_distinct_subset_products(S):
        return []
    
    # Sort composites based on order
    if order == 'asc':
        sorted_comps = sorted(composites)
    elif order == 'desc':
        sorted_comps = sorted(composites, reverse=True)
    elif order == 'fewest_factors':
        sorted_comps = sorted(composites, key=lambda x: len(_prime_factors(x)))
    elif order == 'most_factors':
        sorted_comps = sorted(composites, key=lambda x: len(_prime_factors(x)), reverse=True)
    else:
        sorted_comps = list(composites)
    
    for c in sorted_comps:
        if len(S) >= max_size:
            break
        # Fast reject
        if not _fast_check(S, c):
            continue
        trial = S + [c]
        if _has_distinct_subset_products(trial):
            S = trial
    
    return S

def _local_search(S, all_candidates, max_iter=100, deadline=None):
    """
    Try to improve S by swapping elements or adding/removing.
    Returns improved set if found.
    """
    current = list(S)
    best = list(current)
    all_set = set(all_candidates)
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try removing an element and adding a different one
        for i, elem in enumerate(current):
            if elem not in all_set:
                continue
            trial = current[:i] + current[i+1:]
            # Try adding each candidate not in trial
            for cand in all_candidates:
                if cand in trial:
                    continue
                if not _fast_check(trial, cand):
                    continue
                new_trial = trial + [cand]
                if _has_distinct_subset_products(new_trial):
                    if len(new_trial) > len(best) or (len(new_trial) == len(best) and sorted(new_trial) > sorted(best)):
                        best = new_trial
                        current = new_trial
                        improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best

def _build_max_set():
    """
    Try multiple greedy strategies and local search to find max set.
    """
    primes = [p for p in range(2, 41) if _is_prime(p)]
    composites = [c for c in range(2, 41) if not _is_prime(c)]
    all_candidates = list(range(2, 41))
    
    t0 = time.time()
    deadline = t0 + 80  # Safety margin
    
    # Try multiple greedy orderings
    orders = ['asc', 'desc', 'fewest_factors', 'most_factors']
    best_set = []
    
    for order in orders:
        if time.time() > deadline:
            break
        S = _greedy_build(primes, composites, order=order)
        if len(S) > len(best_set):
            best_set = S
    
    # Also try starting with fewer primes to allow more composites
    # Remove some primes to see if we can add more composites
    prime_subsets_to_try = []
    # Try removing each prime one at a time
    for i in range(len(primes)):
        subset = primes[:i] + primes[i+1:]
        prime_subsets_to_try.append(subset)
    # Try removing pairs of primes
    for i in range(len(primes)):
        for j in range(i+1, len(primes)):
            subset = primes[:i] + primes[i+1:j] + primes[j+1:]
            prime_subsets_to_try.append(subset)
    
    for psubset in prime_subsets_to_try:
        if time.time() > deadline:
            break
        for order in ['asc', 'desc']:
            S = _greedy_build(psubset, composites, order=order)
            if len(S) > len(best_set):
                best_set = S
    
    # Local search on best set found
    if best_set and time.time() < deadline - 5:
        improved = _local_search(best_set, all_candidates, max_iter=50, deadline=deadline)
        if len(improved) > len(best_set):
            best_set = improved
    
    return best_set

def construct():
    """Return up to 18 distinct integers in 2..40 with all subset products distinct."""
    t0 = time.time()
    
    # Primary strategy: Multi-strategy greedy + local search
    S = _build_max_set()
    
    # Ensure list of Python ints
    S = [int(x) for x in S]
    
    # Trim to max 18 if needed
    if len(S) > 18:
        S = S[:18]
        
    # Final verification
    if time.time() - t0 < 85:
        prods = [1]
        for a in S:
            prods = prods + [x * a for x in prods]
        if len(prods) != len(set(prods)):
            # Fallback: return just primes
            S = [p for p in range(2, 41) if _is_prime(p)]
            
    return S
# EVOLVE-BLOCK-END
