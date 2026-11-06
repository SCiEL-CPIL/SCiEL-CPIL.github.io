"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from functools import lru_cache

# EVOLVE-BLOCK-START
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # We need 12 distinct positive integers <= 10000
    # We want to minimize the number of distinct prime factors of all pairwise sums
    
    # Strategy:
    # 1. Precompute primes up to 20000 (since max sum is 10000+9999=19999)
    # 2. Use a greedy/local search approach:
    #    - Start with a set of 12 numbers
    #    - Try to replace elements to reduce the number of distinct prime factors
    
    # First, let's understand the problem better.
    # If we pick numbers that are all multiples of a small number, their sums will also have that factor.
    # For example, if all numbers are even, all sums are even, so 2 is always a factor.
    # But we want to MINIMIZE the number of distinct primes, so we want sums to share common factors.
    
    # Key insight: If we pick numbers from a set where all pairwise sums are divisible by a small set of primes,
    # we minimize the distinct prime count.
    
    # Let's try: pick 12 numbers such that all pairwise sums are divisible by a small set of primes.
    # For example, if all numbers are congruent to 0 mod 2, then all sums are 0 mod 2.
    # If we can also make them 0 mod 3, then all sums are 0 mod 6.
    # But we need 12 distinct numbers <= 10000 that are all 0 mod 6: 6, 12, 18, ..., 72. That's 12 numbers!
    # Sums: 6+12=18, 6+18=24, etc. All sums are divisible by 6.
    # But 18 = 2*3^2, 24 = 2^3*3, etc. The prime factors are {2, 3}. So we have 2 distinct primes.
    
    # Can we do better? Can we get 1 distinct prime?
    # If all sums are powers of 2, that would be ideal. But sums of distinct numbers being powers of 2 is very restrictive.
    # Let's check: if all numbers are powers of 2, sums of two distinct powers of 2 are not powers of 2 (e.g., 2+4=6).
    # So 1 distinct prime is likely impossible.
    
    # Let's try to achieve 2 distinct primes.
    # Approach: pick numbers that are all multiples of 6 (i.e., 2*3).
    # Sums will be multiples of 6, so prime factors are subset of {2, 3} plus possibly others.
    # Wait, 6+12=18=2*3^2, primes are {2,3}. 12+18=30=2*3*5, primes are {2,3,5}. Oops.
    # So multiples of 6 don't guarantee only {2,3} as prime factors.
    
    # Better approach: use local search.
    # Start with a candidate set, evaluate the number of distinct prime factors of all pairwise sums.
    # Then try to improve by replacing one element at a time.
    
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
    
    # Initial candidate: multiples of 6
    # 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72
    best = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
    best_score = evaluate(best)
    
    # Local search: try to replace one element with another to improve the score
    random.seed(42)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all possible replacements
        for i in range(12):
            old_val = best[i]
            # Try replacing with random values
            for _ in range(200):
                if time.time() > deadline:
                    break
                new_val = random.randint(1, 10000)
                if new_val == old_val or new_val in best:
                    continue
                candidate = best.copy()
                candidate[i] = new_val
                score = evaluate(candidate)
                if score < best_score:
                    best = candidate
                    best_score = score
                    improved = True
                    break
    
    return best
# EVOLVE-BLOCK-END
