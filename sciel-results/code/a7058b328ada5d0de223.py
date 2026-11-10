"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

# Primes strictly below 24
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
NUM_PRIMES = len(PRIMES)

# Precompute powers for each prime to limit factorization depth
# We'll factor out all powers of these primes from each number

def extract_small_prime_factor(n):
    """Extract the complete factor of n composed of primes strictly below 24."""
    factor = 1
    for p in PRIMES:
        while n % p == 0:
            factor *= p
            n //= p
    return factor

def extract_factors_for_window(n):
    """Extract small-prime factors for n+1 through n+24. Return list of 24 factors."""
    factors = []
    for i in range(1, 25):
        factors.append(extract_small_prime_factor(n + i))
    return factors

def count_distinct_factors(factors):
    """Count the number of distinct extracted factors."""
    return len(set(factors))

def score_window(n):
    """Return the number of distinct small-prime factors in the window n+1..n+24."""
    factors = extract_factors_for_window(n)
    return count_distinct_factors(factors)

# Precompute a lookup for small numbers? The range is up to 1000000+24, so we can't precompute all.
# But we can precompute for a range and cache.

# Strategy: 
# 1. Do a local search around random starting points.
# 2. Try to minimize the number of distinct factors.
# 
# The minimum possible is 1 (all 24 numbers have the same small-prime factor), but that's unlikely.
# Realistic minimum might be 2-5.
#
# Approach:
# - Use a deadline guard.
# - Start with a coarse scan to find good regions, then refine.
# - Use random sampling + local search.

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s to leave safety margin
    
    best_n = 1000
    best_score = score_window(best_n)
    
    # Phase 1: Coarse scan with random sampling
    # Sample random n values and keep the best
    random.seed(42)
    sample_count = 0
    while time.time() < deadline - 10:
        n = random.randint(1000, 999976)  # n+24 <= 1000000
        s = score_window(n)
        if s < best_score:
            best_score = s
            best_n = n
        sample_count += 1
        if sample_count % 50000 == 0:
            pass  # could log
    
    # Phase 2: Local search around the best found
    # Try neighbors of best_n
    if time.time() < deadline - 5:
        for delta in range(1, 500):
            if time.time() >= deadline - 5:
                break
            for n_candidate in [best_n - delta, best_n + delta]:
                if 1000 <= n_candidate <= 999976:
                    s = score_window(n_candidate)
                    if s < best_score:
                        best_score = s
                        best_n = n_candidate
    
    # Phase 3: More aggressive local search with random perturbations
    if time.time() < deadline - 2:
        for _ in range(2000):
            if time.time() >= deadline - 2:
                break
            # Random perturbation
            perturbation = random.randint(-10000, 10000)
            n_candidate = best_n + perturbation
            if 1000 <= n_candidate <= 999976:
                s = score_window(n_candidate)
                if s < best_score:
                    best_score = s
                    best_n = n_candidate
            else:
                # Try another random point
                n_candidate = random.randint(1000, 999976)
                s = score_window(n_candidate)
                if s < best_score:
                    best_score = s
                    best_n = n_candidate
    
    return best_n

# EVOLVE-BLOCK-END
