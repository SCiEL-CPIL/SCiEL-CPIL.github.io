"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a linear sieve to compute phi(n) for all n up to 1,000,000 and finds the minimum.
    """
    N = 1000000
    phi = list(range(N + 1))
    
    # Linear sieve for Euler's totient
    for i in range(2, N + 1):
        if phi[i] == i: # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
                
    # Find the smallest n with phi(n) % 120 == 0
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
            
    return -1
# EVOLVE-BLOCK-END
