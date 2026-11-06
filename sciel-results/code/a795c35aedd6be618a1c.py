"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute smallest prime factor for fast factorization
    MAX_VAL = 20000
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(n):
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
    
    def evaluate(arr):
        """Return the number of distinct prime factors of all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes |= factorize(s)
        return len(primes)
    
    def get_prime_set(arr):
        """Return the set of distinct prime factors of all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes |= factorize(s)
        return primes
    
    # Strategy: Try to find 12 numbers where all pairwise sums have minimal distinct primes.
    # Key insight: if all numbers are congruent to r mod M, sums are 2r mod M.
    # If M is a prime p and 2r = 0 mod p, then all sums are divisible by p.
    # This gives us at least one common prime factor.
    
    # Let's try a structured approach:
    # Pick numbers from a residue class mod M where M has small prime factors.
    # For example, all numbers ≡ 0 mod 6: sums ≡ 0 mod 6, so 2 and 3 divide all sums.
    # But sums also have other factors.
    
    # Better: try to find numbers where all pairwise sums are "smooth" (few prime factors).
    
    # Initial candidate: multiples of 6
    best = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
    best_score = evaluate(best)
    best_primes = get_prime_set(best)
    
    # Local search with two-phase approach:
    # Phase 1: Try to reduce the number of distinct primes by replacing elements.
    # Phase 2: Fine-tune.
    
    random.seed(42)
    
    # Generate candidate values: focus on values that are multiples of small numbers
    # to increase the chance of shared factors.
    candidate_pool = []
    for m in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30]:
        for k in range(1, 10000 // m + 1):
            val = k * m
            if 1 <= val <= 10000:
                candidate_pool.append(val)
    # Add some random values
    for _ in range(5000):
        candidate_pool.append(random.randint(1, 10000))
    candidate_pool = list(set(candidate_pool))
    random.shuffle(candidate_pool)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all possible replacements
        for i in range(12):
            old_val = best[i]
            # Try replacing with values from the candidate pool
            for new_val in candidate_pool:
                if time.time() > deadline:
                    break
                if new_val == old_val:
                    continue
                if new_val in best:
                    continue
                candidate = best.copy()
                candidate[i] = new_val
                score = evaluate(candidate)
                if score < best_score:
                    best = candidate
                    best_score = score
                    best_primes = get_prime_set(best)
                    improved = True
                    break
    
    return best
# EVOLVE-BLOCK-END
