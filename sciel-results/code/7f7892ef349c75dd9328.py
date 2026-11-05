"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    We use a sieve to compute phi(n) for all n up to 1000000, then find the smallest n
    where phi(n) % 120 == 0.
    """
    N = 1000000
    # Initialize phi array: phi[i] = i
    phi = list(range(N + 1))
    
    # Sieve to compute Euler's totient
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
    
    # Find smallest n with phi(n) divisible by 120
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
    
    # Fallback (should not be reached)
    return N
# EVOLVE-BLOCK-END
