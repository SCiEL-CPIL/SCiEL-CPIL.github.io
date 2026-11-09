"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a highly optimized sieve-like approach to compute totients for all numbers
    up to a limit, then finds the minimum n satisfying the condition.
    """
    # We know n=143 (11*13) works: phi(143) = 143 * (1-1/11) * (1-1/13) = 143 * 10/11 * 12/13 = 120.
    # But let's verify by building a proper totient sieve up to a reasonable bound
    # to be absolutely certain 143 is the minimum, and to have a robust algorithm.
    
    # For efficiency, we only need to check up to a small limit since we expect the answer
    # to be small. Let's go up to 10000 to be safe and fast.
    limit = 10000
    phi = list(range(limit + 1))
    
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
    
    # Find the smallest n >= 1 such that phi[n] % 120 == 0
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
    
    # Fallback: if for some reason not found (shouldn't happen), return 143
    return 143

# EVOLVE-BLOCK-END
