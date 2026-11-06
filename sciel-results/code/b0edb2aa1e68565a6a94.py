"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_val = 1000000 + 24  # 1000024
    
    # Precompute rough_part: largest divisor of m with all prime factors >= 24
    # Use a sieve approach: for each prime p >= 29, divide p out of multiples
    # Actually, we want to compute small_part[m] = m with all prime factors >= 24 removed
    
    # More efficient: compute small_part directly using a modified sieve
    # Initialize small_part[m] = m
    # For each prime p >= 29, divide p out of small_part[multiple] for all multiples
    
    small_part = list(range(max_val + 1))
    
    # Generate primes from 29 to max_val
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    
    # For each prime p >= 29, remove all factors of p from small_part[multiples of p]
    for p in range(29, max_val + 1):
        if is_prime[p]:
            for multiple in range(p, max_val + 1, p):
                while small_part[multiple] % p == 0:
                    small_part[multiple] //= p
    
    # Now small_part[m] contains only primes < 24
    # Search for best n in [1000, 1000000]
    best_n = 1000
    best_count = float('inf')
    
    # Slide window of size 24
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
        
        # Count distinct small_part values for n+1, ..., n+24
        vals = set()
        for i in range(1, 25):
            vals.add(small_part[n + i])
        
        cnt = len(vals)
        if cnt < best_count:
            best_count = cnt
            best_n = n
            if best_count <= 1:
                break  # Can't do better than 1
    
    return best_n
# EVOLVE-BLOCK-END
