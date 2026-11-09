"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 256
    
    # Sieve of Eratosthenes to find primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(math.sqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    # Collect primes
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # The set of primes is a valid solution.
    # For any distinct primes a, b, c, a does not divide b*c because a is prime and a != b, a != c.
    # Size is pi(256) = 55.
    
    # Check time
    if time.time() > deadline:
         # This should never happen, but just in case
         return primes
         
    return primes
# EVOLVE-BLOCK-END
