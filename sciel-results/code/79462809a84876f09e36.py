"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin within 90s limit
    
    N = 1000000
    # We need powerful parts for n and n+1 where n in [10000, 1000000]
    # So we need powerful parts for numbers up to 1000001
    max_val = N + 1
    
    # Sieve to compute powerful part for all integers up to max_val
    # powerful_part[i] = product of p^e where e >= 2 in factorization of i
    
    # Initialize with 1 for all
    pow_part = np.ones(max_val + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes to find primes, then multiply prime powers
    # For each prime p, for each power p^k with k>=2, multiply pow_part at indices divisible by p^k
    # But we need the FULL prime power, so if p^e || n, we multiply by p^e
    
    # Better approach: for each prime p, find the highest power p^e dividing each multiple
    # Actually, let's use a different approach:
    # For each prime p, for each k >= 2, let pk = p^k
    # For each multiple m of pk that is NOT a multiple of p^(k+1), multiply pow_part[m] by p^k
    # This is equivalent to: for each prime p, for each multiple m of p^2, 
    # find the exact exponent e of p in m, and multiply pow_part[m] by p^e
    
    # Simpler: use a modified sieve
    # spf (smallest prime factor) approach might be cleaner but slower
    # Let's use the direct prime power approach
    
    # Find all primes up to max_val using sieve
    is_prime = np.ones(max_val + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(max_val)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # For each prime, compute its powers and update pow_part
    for p in primes:
        p = int(p)
        if p * p > max_val:
            break
        
        # Compute powers p^2, p^3, ... up to max_val
        pk = p * p
        k = 2
        while pk <= max_val:
            # For each multiple of pk, check if it's also divisible by p^(k+1)
            # If not, the exponent of p in that number is exactly k
            # We need to multiply pow_part[m] by p^k for those m
            
            # Multiples of pk: pk, 2*pk, 3*pk, ...
            # Among these, those NOT divisible by p^(k+1) = pk * p
            # i.e., m = j * pk where j is NOT divisible by p
            
            p_next = pk * p  # p^(k+1)
            
            # Indices: pk, 2*pk, 3*pk, ... up to max_val
            # Exclude those that are multiples of p_next
            indices = np.arange(pk, max_val + 1, pk)
            # Filter out multiples of p_next
            mask = (indices % p_next) != 0
            indices = indices[mask]
            
            if len(indices) > 0:
                pow_part[indices] *= pk
            
            pk = p_next
            k += 1
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # Now compute product of consecutive powerful parts
    # n in [10000, 1000000], product = pow_part[n] * pow_part[n+1]
    # We need n such that n >= 10000 and n <= 1000000
    # So we look at indices 10000 to 1000000
    
    if time.time() > deadline - 2.0:
        # Not enough time, return a reasonable default
        return 10000
    
    # Compute products for all valid n
    # n ranges from 10000 to 1000000
    # products[n] = pow_part[n] * pow_part[n+1]
    products = pow_part[10000:1000001] * pow_part[10001:1000002]
    
    # Find the index with maximum product
    max_idx = int(np.argmax(products))
    best_n = 10000 + max_idx
    
    return int(best_n)
# EVOLVE-BLOCK-END
