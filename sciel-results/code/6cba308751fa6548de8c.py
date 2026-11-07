import time
import itertools

P = {'n': 256}

def construct():
    start_time = time.time()
    limit = P['n']
    
    # Find all primes up to limit
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    return primes

# EVOLVE-BLOCK-END
