import time
import math
import random
from itertools import combinations

_MAX_SUM = 20000

# Precompute smallest prime factor
_spf = list(range(_MAX_SUM + 1))
for _i in range(2, int(_MAX_SUM ** 0.5) + 1):
    if _spf[_i] == _i:
        for _j in range(_i * _i, _MAX_SUM + 1, _i):
            if _spf[_j] == _j:
                _spf[_j] = _i

def _factor_set(n):
    s = set()
    while n > 1:
        p = _spf[n]
        s.add(p)
        while n % p == 0:
            n //= p
    return s

# Precompute factor sets for all sums
_pf = [None] * (_MAX_SUM + 1)
for _x in range(2, _MAX_SUM + 1):
    _pf[_x] = _factor_set(_x)

def _score(arr):
    """Number of distinct primes across all pairwise sums."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            primes |= _pf[arr[i] + arr[j]]
    return len(primes)

def _total_foreign_count(arr, allowed_primes):
    """Count how many sums have prime factors outside allowed_primes."""
    total = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            for p in _pf[arr[i] + arr[j]]:
                if p not in allowed_primes:
                    total += 1
    return total

def _anneal(arr, allowed_primes, deadline, rng):
    """
    Simulated annealing to find 12 distinct values such that all pairwise sums
    use only primes in allowed_primes.
    Objective: minimize total_foreign_count (number of "bad" prime occurrences).
    """
    n = len(arr)
    arr = list(arr)
    cur_foreign = _total_foreign_count(arr, allowed_primes)
    best_foreign = cur_foreign
    best_arr = list(arr)
    
    # Candidate pool: numbers whose individual prime factors are mostly in allowed set
    # Actually, we just need sums to be smooth. Let's use a broad pool.
    pool = list(range(1, 10001))
    
    t = 50.0  # initial temperature
    cooling = 0.9995
    
    while time.time() < deadline and t > 0.01:
        i = rng.randint(0, n - 1)
        old_val = arr[i]
        # Pick a new value
        new_val = pool[rng.randint(0, len(pool) - 1)]
        if new_val == old_val or new_val in arr:
            t *= cooling
            continue
        
        # Compute change in foreign count
        delta = 0
        for j in range(n):
            if j == i:
                continue
            s_old = old_val + arr[j]
            s_new = new_val + arr[j]
            for p in _pf[s_old]:
                if p not in allowed_primes:
                    delta -= 1
            for p in _pf[s_new]:
                if p not in allowed_primes:
                    delta += 1
        
        if delta <= 0 or rng.random() < math.exp(-delta / t):
            arr[i] = new_val
            cur_foreign += delta
            if cur_foreign < best_foreign:
                best_foreign = cur_foreign
                best_arr = list(arr)
        
        t *= cooling
    
    return best_foreign, best_arr

def _try_prime_set(prime_set, deadline, rng):
    """
    Given a set of allowed primes, try to find 12 distinct integers <= 10000
    such that all pairwise sums have only these prime factors.
    Returns (score, arr) or (None, None) if not found.
    """
    if not prime_set:
        return None, None
    
    # Generate candidate pool: numbers such that for each pair (a,b) in the set,
    # a+b should only have prime factors in prime_set.
    # Strategy: start with random assignments, then anneal.
    
    # Initial guess: pick small numbers
    arr = list(range(1, 13))
    # But we need them to be "smooth" w.r.t. the allowed primes.
    # Let's try: pick numbers that are products of allowed primes.
    # For simplicity, just start random and anneal.
    
    best_foreign, best_arr = _anneal(arr, prime_set, deadline, rng)
    
    if best_foreign == 0:
        # Verify all sums use only allowed primes
        n = len(best_arr)
        for i in range(n):
            for j in range(i + 1, n):
                for p in _pf[best_arr[i] + best_arr[j]]:
                    if p not in prime_set:
                        return None, None
        return len(prime_set), best_arr
    
    # If not perfect, still return the best we found (score = actual score)
    actual_score = _score(best_arr)
    return actual_score, best_arr

def construct():
    start = time.time()
    deadline = start + 85.0
    rng = random.Random(42)
    
    n = 12
    best_score = 10**9
    best_arr = None
    
    # List of small primes to try
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Try prime sets of increasing size
    # First, try very small sets (2-4 primes)
    for size in range(2, 6):
        if time.time() > deadline - 5.0:
            break
        # Sample some combinations
        num_trials = 20 if size <= 3 else 10
        for _ in range(num_trials):
            if time.time() > deadline - 3.0:
                break
            prime_set = set(rng.sample(small_primes, size))
            # Always include 2 if possible for flexibility
            if 2 not in prime_set and rng.random() < 0.7:
                prime_set.add(2)
            
            sub_deadline = min(deadline - 2.0, time.time() + 5.0)
            sc, arr = _try_prime_set(prime_set, sub_deadline, rng)
            if sc is not None and sc < best_score:
                best_score = sc
                best_arr = list(arr)
    
    # If we found a perfect solution, return it
    if best_arr is not None:
        return best_arr
    
    # Fallback: local search from good seeds
    seeds = [
        list(range(6, 6 * n + 1, 6)),
        list(range(12, 12 * n + 1, 12)),
        list(range(1, n + 1)),
    ]
    
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
        sc = _score(seed)
        if sc < best_score:
            best_score = sc
            best_arr = list(seed)
    
    if best_arr is None:
        return list(range(1, 13))
    return best_arr
