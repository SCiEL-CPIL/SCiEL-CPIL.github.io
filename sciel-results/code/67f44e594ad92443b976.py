"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = 60.0  # Safety margin
    
    # Sieve of Eratosthenes to find primes up to 256
    limit = 256
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    result = [1]
    
    for p in primes:
        # Find largest power of p <= 256
        max_val = p
        current = p
        while current * p <= limit:
            current *= p
        result.append(current)
        
        # Check time
        if time.time() - start_time > deadline - 5:
            break
            
    # Ensure distinctness (should be guaranteed by prime power property, but just in case)
    # The set of prime powers for distinct primes are always distinct and coprime.
    # 1 is coprime to all.
    
    return result
# EVOLVE-BLOCK-END
