"""Rewritten program: model the problem as a constraint satisfaction problem on prime-exponent vectors."""
import time
import math
import random

# EVOLVE-BLOCK-START

P = {'n': 40, 'maximum_size': 18}


def _primes_upto(n):
    """Return list of primes <= n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]


def _factorize(x, primes):
    """Return dict mapping prime to exponent for x, given list of primes."""
    f = {}
    for p in primes:
        if p * p > x:
            break
        if x % p == 0:
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            f[p] = e
    if x > 1:
        f[x] = 1
    return f


def _exponent_vector(x, primes):
    """Return tuple of exponents for x with respect to given primes."""
    f = _factorize(x, primes)
    return tuple(f.get(p, 0) for p in primes)


def _subset_products_distinct(nums, primes):
    """
    Check whether all 2^k subset products of nums are distinct.
    Uses exponent-vector sums: two subsets have equal product iff their
    exponent-vector sums are equal.
    """
    k = len(nums)
    if k == 0:
        return True
    vectors = [_exponent_vector(n, primes) for n in nums]
    
    # Generate all subset sum vectors; if any duplicate, not distinct.
    seen = {tuple([0] * len(vectors[0]))}  # empty subset
    for mask in range(1, 1 << k):
        # Compute exponent sum for this subset
        vec = [0] * len(vectors[0])
        for i in range(k):
            if (mask >> i) & 1:
                v = vectors[i]
                for j in range(len(v)):
                    vec[j] += v[j]
        t = tuple(vec)
        if t in seen:
            return False
        seen.add(t)
    return True


def _try_add(nums, candidate, primes, deadline):
    """Try adding candidate to nums; return True if still distinct."""
    if time.time() > deadline:
        return False
    new_nums = nums + [candidate]
    return _subset_products_distinct(new_nums, primes)


def _greedy_select(candidates, primes, deadline):
    """
    Greedy: repeatedly pick the candidate that allows the largest future set,
    using a forward-checking heuristic.
    """
    selected = []
    remaining = list(candidates)
    
    while remaining and time.time() < deadline:
        best_idx = -1
        best_score = -1
        for i, cand in enumerate(remaining):
            # Heuristic: prefer smaller numbers, but check feasibility
            if _try_add(selected, cand, primes, deadline):
                # Score: prefer candidates that keep more options open
                # Simple heuristic: smaller number is better (more primes available)
                score = -cand
                if score > best_score:
                    best_score = score
                    best_idx = i
        if best_idx == -1:
            break
        selected.append(remaining[best_idx])
        remaining.pop(best_idx)
    return selected


def _local_search(nums, primes, deadline):
    """
    Local search: try swapping elements to improve the set.
    Returns a possibly improved list.
    """
    if time.time() > deadline:
        return nums
    
    best = list(nums)
    best_len = len(best)
    candidates = list(range(2, P['n'] + 1))
    
    for _ in range(100):  # limited iterations
        if time.time() > deadline:
            break
        improved = False
        # Try adding a new element
        for cand in candidates:
            if cand in best:
                continue
            if _try_add(best, cand, primes, deadline):
                new_best = best + [cand]
                if len(new_best) > best_len:
                    best = new_best
                    best_len = len(best)
                    improved = True
                    break
        if improved:
            continue
        # Try swapping
        for i in range(len(best)):
            if time.time() > deadline:
                break
            for cand in candidates:
                if cand in best:
                    continue
                # Remove best[i], add cand
                trial = best[:i] + best[i+1:] + [cand]
                if _subset_products_distinct(trial, primes):
                    # Accept swap if it doesn't reduce size (same size, but maybe better position)
                    # For now, only accept if it enables future growth
                    pass
        if not improved:
            break
    return best


def construct():
    start = time.time()
    deadline = start + 50.0  # 50s safety margin
    
    primes = _primes_upto(P['n'])
    candidates = list(range(2, P['n'] + 1))
    
    # Step 1: Greedy selection
    selected = _greedy_select(candidates, primes, deadline)
    
    # Step 2: Local search to improve
    selected = _local_search(selected, primes, deadline)
    
    # Ensure all elements are valid Python ints
    result = [int(x) for x in selected]
    result.sort()
    
    return result

# EVOLVE-BLOCK-END
