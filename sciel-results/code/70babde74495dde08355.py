"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import isqrt

# EVOLVE-BLOCK-START

def _factorize(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.add(n)
    return factors

def _count_distinct_primes_in_sums(arr):
    """Count distinct prime factors across all pairwise sums of distinct elements."""
    prime_set = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            prime_set |= _factorize(s)
    return len(prime_set)

def _try_multiples_of(base, n=12, max_val=10000, max_iter=500):
    """Try to find n distinct multiples of base that minimize prime factors in sums."""
    best_val = float('inf')
    best_arr = None
    candidates = list(range(base, max_val + 1, base))
    if len(candidates) < n:
        return None, float('inf')
    
    # Try a few random subsets
    for _ in range(max_iter):
        arr = sorted(random.sample(candidates, n))
        val = _count_distinct_primes_in_sums(arr)
        if val < best_val:
            best_val = val
            best_arr = arr[:]
            if best_val <= 3:
                break
    return best_arr, best_val

def _local_search(arr, max_iter=200):
    """Local search: try replacing one element with another to reduce prime count."""
    best_val = _count_distinct_primes_in_sums(arr)
    best_arr = arr[:]
    n = len(arr)
    
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            # Try replacing arr[i] with a nearby value
            for delta in range(1, 20):
                new_val = best_arr[i] + delta
                if new_val <= 10000 and new_val not in best_arr:
                    test_arr = best_arr[:]
                    test_arr[i] = new_val
                    val = _count_distinct_primes_in_sums(test_arr)
                    if val < best_val:
                        best_val = val
                        best_arr = test_arr
                        improved = True
                        break
                new_val = best_arr[i] - delta
                if new_val >= 1 and new_val not in best_arr:
                    test_arr = best_arr[:]
                    test_arr[i] = new_val
                    val = _count_distinct_primes_in_sums(test_arr)
                    if val < best_val:
                        best_val = val
                        best_arr = test_arr
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    return best_arr, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    best_arr = list(range(1, 13))
    best_val = _count_distinct_primes_in_sums(best_arr)
    
    # Try multiples of small numbers
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 32, 36, 40, 48, 60]:
        if time.time() > deadline:
            break
        arr, val = _try_multiples_of(base)
        if arr is not None and val < best_val:
            best_val = val
            best_arr = arr
            # Try local search
            if time.time() < deadline:
                arr2, val2 = _local_search(best_arr, max_iter=100)
                if val2 < best_val:
                    best_val = val2
                    best_arr = arr2
    
    # Try random search with small numbers
    if time.time() < deadline:
        for _ in range(200):
            if time.time() > deadline:
                break
            # Pick 12 random numbers from 1..1000
            arr = sorted(random.sample(range(1, 1001), 12))
            val = _count_distinct_primes_in_sums(arr)
            if val < best_val:
                best_val = val
                best_arr = arr
                if time.time() < deadline:
                    arr2, val2 = _local_search(best_arr, max_iter=50)
                    if val2 < best_val:
                        best_val = val2
                        best_arr = arr2
    
    # Try structured sets: all even numbers, all multiples of 3, etc.
    structured_sets = [
        list(range(2, 25, 2)),       # Even numbers 2..24
        list(range(3, 40, 3)),       # Multiples of 3
        list(range(4, 50, 4)),       # Multiples of 4
        list(range(6, 73, 6)),       # Multiples of 6
        list(range(12, 145, 12)),    # Multiples of 12
        list(range(24, 289, 24)),    # Multiples of 24
        list(range(1, 13, 1)),       # 1..12
        list(range(1, 13, 1)),       # Same as above
    ]
    
    for s in structured_sets:
        if time.time() > deadline:
            break
        if len(set(s)) != 12 or any(x > 10000 or x < 1 for x in s):
            continue
        val = _count_distinct_primes_in_sums(s)
        if val < best_val:
            best_val = val
            best_arr = s[:]
    
    return best_arr

# EVOLVE-BLOCK-END
