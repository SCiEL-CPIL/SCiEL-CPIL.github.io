import time
from math import gcd

def construct():
    # The set of all primes in [2, 256] is a valid solution.
    # Size is 54.
    # Any set of pairwise coprime integers is valid.
    # Primes are the smallest pairwise coprime integers.
    # It is conjectured (and likely true) that this is the maximum cardinality.
    
    # Sieve of Eratosthenes
    N = 256
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Optional: Quick local search to see if we can improve
    # But given the constraint strength, primes are likely optimal.
    # We return the primes.
    
    return primes
