import time

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a sieve to compute phi(n) for all n up to 1,000,000.
    """
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    N = 1000000
    target = 120
    
    # Initialize phi[i] = i
    phi = list(range(N + 1))
    
    # Sieve of Eratosthenes variant for totient
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
                
    # Find the smallest n such that phi(n) % 120 == 0
    # We start from 1. Note phi(1) = 1, which is not divisible by 120.
    for n in range(1, N + 1):
        if phi[n] % target == 0:
            return n
            
    # Fallback if no solution found (should not happen)
    return 1

# EVOLVE-BLOCK-END
