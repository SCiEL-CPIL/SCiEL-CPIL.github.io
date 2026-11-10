"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Strategy:
    1. Precompute smallest prime factor (SPF) up to 1,000,000 using a sieve.
    2. Compute phi(n) for all n using the SPF table.
    3. Find the smallest n such that phi(n) % 120 == 0.
    """
    LIMIT = 1000000
    
    # Sieve for smallest prime factor
    spf = list(range(LIMIT + 1))
    for i in range(2, int(LIMIT**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute phi for all n
    phi = list(range(LIMIT + 1))
    for i in range(2, LIMIT + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, LIMIT + 1, i):
                phi[j] -= phi[j] // i
    
    # Find smallest n with phi(n) divisible by 120
    for n in range(1, LIMIT + 1):
        if phi[n] % 120 == 0:
            return n
    
    # Fallback (should not happen)
    return LIMIT
# EVOLVE-BLOCK-END
