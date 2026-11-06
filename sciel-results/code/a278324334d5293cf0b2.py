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
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    # Deadline guard: 90s limit, keep 5s margin
    deadline = start_time + 85.0
    
    # Range for n: [10000, 999999] so n+1 <= 1000000
    # We need powerful parts for numbers up to 1000000.
    N_MAX_VAL = 1000000
    N_MIN = 10000
    N_MAX = N_MAX_VAL - 1 # n can be up to 999999
    
    # Initialize powerful part array with 1s
    pp = np.ones(N_MAX_VAL + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes to get primes
    is_prime = np.ones(N_MAX_VAL + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N_MAX_VAL)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # For each prime, update pp for its powers
    for p in primes:
        p2 = p * p
        if p2 > N_MAX_VAL:
            break
            
        # Iterate over powers p^k for k >= 2
        power = p2
        k = 2
        while power <= N_MAX_VAL:
            # Indices divisible by power
            # We want to multiply pp[n] by power for all n such that v_p(n) == k
            # This means n is divisible by p^k but NOT by p^(k+1)
            
            # Create mask for multiples of power
            # Instead of full boolean array, use slicing if possible?
            # But we need to exclude multiples of next power.
            
            # Let's use boolean masks for clarity, they are fast enough for 1e6
            mask_current = np.zeros(N_MAX_VAL + 1, dtype=bool)
            mask_current[0:power+1:power] = True # 0 is not in range of interest for pp>1 usually, but pp[0]=1
            
            # If there is a next power, exclude its multiples
            next_power = power * p
            if next_power <= N_MAX_VAL:
                mask_next = np.zeros(N_MAX_VAL + 1, dtype=bool)
                mask_next[0:next_power+1:next_power] = True
                target_mask = mask_current & ~mask_next
            else:
                target_mask = mask_current
            
            # Apply update
            # pp[target_mask] *= power
            # Note: pp is int64. power fits in int64.
            # The product might overflow?
            # Max pp is product of prime powers.
            # n <= 1e6. Max pp(n) <= n.
            # So pp[n] <= 1e6.
            # pp[n] * pp[n+1] <= 1e12. Fits in int64 (9e18).
            # Intermediate pp[n] values are <= n, so no overflow in accumulation.
            
            pp[target_mask] *= power
            
            power = next_power
            k += 1
            
            # Time check
            if time.time() > deadline - 10:
                break
        
        if time.time() > deadline - 10:
            break
            
    # Find max product
    # We only care about n in [N_MIN, N_MAX]
    # products = pp[N_MIN:N_MAX+1] * pp[N_MIN+1:N_MAX+2]
    # This creates a view or copy.
    # Let's do it to be safe and fast.
    
    pp_slice = pp[N_MIN:N_MAX+2]
    products = pp_slice[:-1] * pp_slice[1:]
    
    # Find index of max
    max_idx = np.argmax(products)
    best_n = N_MIN + max_idx
    
    return int(best_n)

# EVOLVE-BLOCK-END
