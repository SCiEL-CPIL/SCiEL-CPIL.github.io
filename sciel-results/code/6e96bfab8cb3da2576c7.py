"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import isqrt
from functools import lru_cache

# EVOLVE-BLOCK-START

def _prime_factors(n):
    """Return set of prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _distinct_primes_in_sums(arr):
    """Count distinct prime factors across all pairwise sums of distinct elements."""
    n = len(arr)
    all_primes = set()
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            all_primes.update(_prime_factors(s))
    return len(all_primes)

def _small_primes(limit):
    """Generate primes up to limit using sieve."""
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def _random_candidate(primes, max_val, n=12):
    """Generate a random set of n distinct positive integers <= max_val,
    biased toward multiples of small primes to reduce prime factor diversity."""
    # Try to pick numbers that are multiples of a small subset of primes
    # Use a few small primes as "building blocks"
    base_primes = primes[:8]  # 2,3,5,7,11,13,17,19
    chosen = set()
    attempts = 0
    while len(chosen) < n and attempts < 500:
        attempts += 1
        # Pick a random small prime and a random multiple
        p = random.choice(base_primes)
        max_mult = max_val // p
        if max_mult < 1:
            continue
        k = random.randint(1, max_mult)
        val = p * k
        if val <= max_val and val > 0:
            chosen.add(val)
    # Fill remaining with random numbers
    while len(chosen) < n:
        val = random.randint(1, max_val)
        chosen.add(val)
    return list(chosen)

def _local_search(arr, primes, max_val, time_limit, rng):
    """Local search: try replacing one element at a time to reduce prime count."""
    n = len(arr)
    current_score = _distinct_primes_in_sums(arr)
    best_arr = list(arr)
    best_score = current_score
    start_time = time.time()
    
    # Precompute for faster evaluation
    while time.time() - start_time < time_limit:
        improved = False
        # Try single-element replacements
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            # Try several candidate replacements
            for _ in range(50):
                # Generate a candidate: multiple of a small prime
                p = random.choice(primes[:10])
                max_mult = max_val // p
                if max_mult < 1:
                    continue
                k = random.randint(1, max_mult)
                new_val = p * k
                if new_val <= max_val and new_val not in best_arr:
                    trial = list(best_arr)
                    trial[i] = new_val
                    score = _distinct_primes_in_sums(trial)
                    if score < best_score:
                        best_arr = trial
                        best_score = score
                        improved = True
                        break
        if not improved:
            # Try random restarts of local moves
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                for _ in range(20):
                    new_val = random.randint(1, max_val)
                    if new_val not in best_arr:
                        trial = list(best_arr)
                        trial[i] = new_val
                        score = _distinct_primes_in_sums(trial)
                        if score < best_score:
                            best_arr = trial
                            best_score = score
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                # Stuck; break out
                break
    
    return best_arr, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    n = 12
    max_val = 10000
    rng = random.Random(42)
    
    primes = _small_primes(max_val)
    
    # Strategy 1: Start with structured candidates
    best_arr = None
    best_score = float('inf')
    
    # Try several random structured starts
    num_starts = 30
    per_start_time = 80.0 / num_starts  # ~2.67 seconds per start
    
    for s in range(num_starts):
        if time.time() > deadline - 2:
            break
        
        # Generate initial candidate
        arr = _random_candidate(primes, max_val, n)
        
        # Ensure distinctness (should be from set, but double-check)
        if len(set(arr)) != n:
            continue
        
        # Quick evaluation
        score = _distinct_primes_in_sums(arr)
        if score < best_score:
            best_score = score
            best_arr = list(arr)
        
        # Local search with time budget
        remaining = deadline - 2 - time.time()
        if remaining < 0.5:
            break
        arr2, score2 = _local_search(arr, primes, max_val, remaining, rng)
        if score2 < best_score:
            best_score = score2
            best_arr = arr2
    
    # Fallback: if best_arr is None, use a simple structured set
    if best_arr is None:
        # Use multiples of 2: 2,4,6,...,24
        best_arr = [2 * i for i in range(1, 13)]
    
    # Final validation
    assert len(best_arr) == n
    assert len(set(best_arr)) == n
    assert all(1 <= x <= max_val for x in best_arr)
    
    return best_arr

# EVOLVE-BLOCK-END
