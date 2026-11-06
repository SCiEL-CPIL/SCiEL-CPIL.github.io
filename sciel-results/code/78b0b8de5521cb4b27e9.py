"""Program to minimize distinct prime factors across pairwise sums."""
import random
import math
from itertools import combinations

def _sieve_smallest_prime(limit):
    """Return list spf where spf[i] is smallest prime factor of i."""
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _prime_factors(n, spf):
    """Return set of prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _count_distinct_primes(arr, spf):
    """Count distinct primes dividing all pairwise sums of arr."""
    prime_set = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            s = arr[i] + arr[j]
            n = s
            while n > 1:
                p = spf[n]
                prime_set.add(p)
                while n % p == 0:
                    n //= p
    return len(prime_set)

def construct():
    n = 12
    max_val = 10000
    
    # Precompute smallest prime factor up to 20000 (max sum of two <= 10000)
    limit = 2 * max_val
    spf = _sieve_smallest_prime(limit)
    
    # Generate candidate values: multiples of 6 are a good start
    # because sums of multiples of 6 are divisible by 6 (primes 2, 3)
    candidates_pool = list(range(6, max_val + 1, 6))
    
    # Initial configurations
    initial_candidates = [
        candidates_pool[:n],  # 6, 12, ..., 72
        candidates_pool[:n],  # Same
        [6*i for i in range(1, n+1)],
        [30*i for i in range(1, n+1)],
    ]
    
    # Add some random initial candidates
    random.seed(42)
    for _ in range(10):
        initial_candidates.append(random.sample(candidates_pool, n))
        
    best_score = float('inf')
    best_arr = None
    
    for cand in initial_candidates:
        if len(set(cand)) != n or any(x < 1 or x > max_val for x in cand):
            continue
        score = _count_distinct_primes(cand, spf)
        if score < best_score:
            best_score = score
            best_arr = list(cand)
    
    if best_arr is None:
        best_arr = [6*i for i in range(1, n+1)]
        best_score = _count_distinct_primes(best_arr, spf)
    
    # Local search
    # We will try to replace each element with another multiple of 6
    # to see if it reduces the prime count.
    
    improved = True
    max_epochs = 50
    
    for epoch in range(max_epochs):
        if not improved:
            break
        improved = False
        
        for i in range(n):
            old_val = best_arr[i]
            
            # Try all other multiples of 6
            # To speed up, we can skip if the set of primes doesn't change?
            # But calculating the set is the expensive part.
            
            for new_val in candidates_pool:
                if new_val == old_val:
                    continue
                if new_val in best_arr:
                    continue
                
                trial = list(best_arr)
                trial[i] = new_val
                
                new_score = _count_distinct_primes(trial, spf)
                
                if new_score < best_score:
                    best_score = new_score
                    best_arr = trial
                    old_val = new_val # Update old_val for subsequent checks? 
                    # Actually, we broke the uniqueness check if we don't restart?
                    # The `new_val in best_arr` check uses the UPDATED best_arr?
                    # No, `best_arr` is updated. So `new_val` is now in `best_arr`.
                    # So subsequent `new_val` checks will correctly skip if they collide.
                    improved = True
                    # We don't break immediately to allow further improvements in the same epoch?
                    # Or break to re-evaluate with the new value?
                    # Usually, greedy local search accepts the first improvement and restarts or continues.
                    # Let's continue scanning for this index with the new base.
                    # But `candidates_pool` is static.
                    # We should probably break and restart the index loop to ensure we are always comparing against the current best.
                    # However, simple hill climbing: accept and move to next index.
                    pass
            
            # Optional: Random restarts if stuck?
            
    return best_arr
