"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 50
    
    N = 256
    
    # Generate all primes in [2, N]
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(N)
    
    # Try to see if we can improve by swapping some primes for composites
    # For now, return primes as a strong baseline
    # This should be optimal or near-optimal
    
    return primes
# EVOLVE-BLOCK-END
