import math

def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    We perform a direct sieve-based computation of phi(n) for n in [1, N] and return
    the smallest n for which phi(n) % 120 == 0. This is fast (O(N log log N)) and
    guarantees correctness and minimality within the range.
    """
    N = 1000000
    # Initialize phi[i] = i
    phi = list(range(N + 1))
    
    # Sieve of Eratosthenes-style update for phi
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
    
    # Find smallest n with phi(n) divisible by 120
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
    
    # Should not reach here, but return a fallback
    return N
