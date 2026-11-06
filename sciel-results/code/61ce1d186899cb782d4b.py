"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import product

def construct():
    """
    Find n in [1000, 1000000] such that for each n+i (1<=i<=24),
    the factor composed of primes strictly below 24 has the minimum
    number of distinct prime factors.
    
    Primes strictly below 24: 2, 3, 5, 7, 11, 13, 17, 19, 23
    
    For each n+i, we extract the largest factor of n+i that is composed
    only of these primes. Then we count the distinct primes in all 24 such factors.
    We want to minimize this count.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    def extract_small_prime_factor(m):
        """Extract the complete factor of m composed of primes strictly below 24."""
        result = 1
        for p in small_primes:
            while m % p == 0:
                result *= p
                m //= p
        return result
    
    def count_distinct_factors(n):
        """Count distinct primes in the small-prime factors of n+1 to n+24."""
        distinct = set()
        for i in range(1, 25):
            factor = extract_small_prime_factor(n + i)
            # Extract distinct primes from this factor
            temp = factor
            for p in small_primes:
                while temp % p == 0:
                    distinct.add(p)
                    temp //= p
        return len(distinct)
    
    # First, try to find a good solution quickly
    best_n = 1000
    best_score = count_distinct_factors(best_n)
    
    # Search for better solutions
    # Strategy: Look for n where n+1 to n+24 have few small prime factors
    # This happens when the numbers are "smooth" with respect to large primes
    
    # Try some heuristic approaches
    # 1. Random search
    import random
    random.seed(42)
    
    # Try a range of values
    for n in range(1000, 1000001, 100):
        if time.time() > deadline:
            break
        score = count_distinct_factors(n)
        if score < best_score:
            best_score = score
            best_n = n
    
    # Fine-tune around the best found
    if best_n > 1000:
        search_start = max(1000, best_n - 200)
        search_end = min(1000000, best_n + 200)
        for n in range(search_start, search_end + 1):
            if time.time() > deadline:
                break
            score = count_distinct_factors(n)
            if score < best_score:
                best_score = score
                best_n = n
    
    # Try to find even better solutions with targeted search
    # Look for n where many of n+i are powers of 2, 3, etc.
    if time.time() < deadline - 10:
        # Try specific patterns
        for n in range(1000, 1000001, 7):
            if time.time() > deadline:
                break
            score = count_distinct_factors(n)
            if score < best_score:
                best_score = score
                best_n = n
                # Continue searching for even better
                if best_score <= 3:
                    break
    
    # Final fine-tuning
    if best_score > 2 and time.time() < deadline - 5:
        for n in range(max(1000, best_n - 50), min(1000000, best_n + 50) + 1):
            if time.time() > deadline:
                break
            score = count_distinct_factors(n)
            if score < best_score:
                best_score = score
                best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
