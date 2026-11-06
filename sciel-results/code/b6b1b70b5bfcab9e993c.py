"""
Strategy: Exhaustive Precomputation via Sieve.
Instead of searching for specific candidates using CRT or random sampling, we precompute the powerful part
for every integer in the valid range [10000, 1000000].
We use a modified sieve to accumulate prime powers p^e (e>=2) into an array.
Then we simply scan the array for the maximum product of consecutive powerful parts.
This guarantees finding the global optimum within the range, provided the sieve is efficient.
"""

import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill
    
    limit = 1000000
    min_n = 10000
    
    # Initialize powerful part array
    pp = [1] * (limit + 1)
    
    # Sieve of Eratosthenes to identify primes
    # We only need primes up to limit
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    # Precompute powerful parts
    # For each prime p, for each power p^e (e>=2), mark multiples
    for p in range(2, limit + 1):
        if not is_prime[p]:
            continue
        
        # Check time
        if time.time() > deadline:
            break
            
        power = p * p
        while power <= limit:
            next_power = power * p
            
            # For each multiple m of power
            # If m is not divisible by next_power, then p^e is the exact highest power of p dividing m
            # And since e >= 2, we multiply pp[m] by power
            for m in range(power, limit + 1, power):
                if m % next_power != 0:
                    pp[m] *= power
                    
            power = next_power
            
    # Find the maximum product of consecutive powerful parts in [10000, 999999]
    best_n = min_n
    best_score = pp[min_n] * pp[min_n + 1]
    
    for i in range(min_n, limit):
        score = pp[i] * pp[i + 1]
        if score > best_score:
            best_score = score
            best_n = i
            
        # Check time just in case, though this loop is fast
        if time.time() > deadline:
            break
            
    return best_n

# EVOLVE-BLOCK-END
