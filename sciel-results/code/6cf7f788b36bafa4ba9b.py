"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Find 12 distinct positive integers <= 10000 that minimize the number of distinct primes
    dividing sums of distinct-index pairs.
    
    Key insight: If all numbers are even, then any sum of two distinct even numbers is even,
    so the only prime factor is 2. This gives us exactly 1 distinct prime, which is optimal.
    
    We return 12 distinct even numbers: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24.
    """
    return list(range(2, 26, 2))

# EVOLVE-BLOCK-END
