"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import Counter

# EVOLVE-BLOCK-START

def _score_set_fast(arr, smallest_prime):
    """Fast scoring using precomputed smallest prime factor."""
    prime_set = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i+1, n):
            s = ai + arr[j]
            if s > 1:
                m = s
                while m > 1:
                    p = smallest_prime[m]
                    prime_set.add(p)
                    while m % p == 0:
                        m //= p
    return len(prime_set)

def _score_reduced(arr_reduced, smallest_prime):
    """Score the reduced problem: count distinct prime factors of pairwise sums of arr_reduced."""
    prime_set = set()
    n = len(arr_reduced)
    for i in range(n):
        ai = arr_reduced[i]
        for j in range(i+1, n):
            s = ai + arr_reduced[j]
            if s > 1:
                m = s
                while m > 1:
                    p = smallest_prime[m]
                    prime_set.add(p)
                    while m % p == 0:
                        m //= p
    return len(prime_set)

def _num_prime_factors(d, smallest_prime):
    """Count distinct prime factors of d."""
    cnt = set()
    m = d
    while m > 1:
        p = smallest_prime[m]
        cnt.add(p)
        while m % p == 0:
            m //= p
    return len(cnt)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s safety margin
    
    # Precompute smallest prime factor up to 20000
    MAX_VAL = 20000
    smallest_prime = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if smallest_prime[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if smallest_prime[j] == j:
                    smallest_prime[j] = i
    
    random.seed(42)
    
    best_set = None
    best_score = float('inf')
    
    # Strategy 1: Small consecutive even numbers
    candidates = list(range(2, 10000, 2))
    initial = candidates[:12]
    score = _score_set_fast(initial, smallest_prime)
    if score < best_score:
        best_score = score
        best_set = initial[:]
    
    # Strategy 2: Multiples of 6
    candidates6 = list(range(6, 10000, 6))
    initial6 = candidates6[:12]
    score6 = _score_set_fast(initial6, smallest_prime)
    if score6 < best_score:
        best_score = score6
        best_set = initial6[:]
    
    # Strategy 3: Systematic search over common divisors d
    # Key insight: if all numbers are multiples of d, then all pairwise sums
    # are multiples of d. The total distinct primes = primes(d) + primes of
    # (a_i/d + a_j/d) for all pairs.
    
    # Candidate d values: products of small primes, up to ~800
    candidate_ds = []
    for d in range(2, 834):
        if 12 * d <= 10000:
            candidate_ds.append(d)
    
    # Sort by number of prime factors of d (prefer fewer)
    candidate_ds.sort(key=lambda d: _num_prime_factors(d, smallest_prime))
    
    # Try top candidates for d
    for d in candidate_ds[:300]:
        if time.time() > deadline:
            break
        
        max_reduced = 10000 // d
        if max_reduced < 12:
            continue
        
        # Initial: smallest 12 reduced values
        initial_reduced = list(range(1, 13))
        if initial_reduced[-1] > max_reduced:
            continue
        
        # Score this initial
        d_primes = _num_prime_factors(d, smallest_prime)
        cur_score = _score_reduced(initial_reduced, smallest_prime) + d_primes
        if cur_score < best_score:
            best_score = cur_score
            best_set = [x * d for x in initial_reduced]
        
        # Local search on reduced values
        improved = True
        local_best = initial_reduced[:]
        local_score = cur_score
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                old = local_best[i]
                # Try nearby values
                for new_r in range(max(1, old - 20), min(max_reduced + 1, old + 21)):
                    if new_r == old:
                        continue
                    if new_r in local_best:
                        continue
                    test = local_best[:]
                    test[i] = new_r
                    test.sort()
                    ns = _score_reduced(test, smallest_prime) + d_primes
                    if ns < local_score:
                        local_score = ns
                        local_best = test
                        improved = True
                        break
        
        if local_score < best_score:
            best_score = local_score
            best_set = [x * d for x in local_best]
    
    # Strategy 4: Random search with local improvement
    for trial in range(30):
        if time.time() > deadline:
            break
        # Random initial set
        rand_set = random.sample(range(1, 10001), 12)
        rand_set.sort()
        s = _score_set_fast(rand_set, smallest_prime)
        if s < best_score:
            best_score = s
            best_set = rand_set[:]
        
        # Local search: try swapping one element
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                old = best_set[i]
                # Try replacing with a random value
                for _ in range(30):
                    new_val = random.randint(1, 10000)
                    if new_val in best_set and new_val != old:
                        continue
                    test_set = best_set[:]
                    test_set[i] = new_val
                    test_set.sort()
                    ns = _score_set_fast(test_set, smallest_prime)
                    if ns < best_score:
                        best_score = ns
                        best_set = test_set
                        improved = True
                        break
    
    # More systematic local search on the best set
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                old = best_set[i]
                # Try nearby values
                for offset in range(-100, 101):
                    if offset == 0:
                        continue
                    new_val = old + offset
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in best_set:
                        continue
                    test_set = best_set[:]
                    test_set[i] = new_val
                    test_set.sort()
                    ns = _score_set_fast(test_set, smallest_prime)
                    if ns < best_score:
                        best_score = ns
                        best_set = test_set
                        improved = True
                        break
    
    return best_set

# EVOLVE-BLOCK-END
