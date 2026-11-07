"""Rewritten program: maximize number of distinct integers in 2..40 with distinct subset products."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns a list of integers in 2..40 with distinct subset products.
    Strategy: Select powers of primes such that for each prime p, the set of
    exponents of p in the selected numbers has distinct subset sums.
    This ensures global distinctness of subset products.
    
    Primes in 2..40: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    Max powers <= 40:
    p=2: 2, 4, 8, 16, 32 (exponents 1, 2, 3, 4, 5). Max distinct subset sum set size: 3 (e.g., {1, 2, 4})
    p=3: 3, 9, 27 (exponents 1, 2, 3). Max distinct subset sum set size: 2 (e.g., {1, 2})
    p=5: 5, 25 (exponents 1, 2). Max distinct subset sum set size: 2 (e.g., {1, 2})
    p=7: 7 (exponent 1). Size 1.
    p=11: 11. Size 1.
    p=13: 13. Size 1.
    p=17: 17. Size 1.
    p=19: 19. Size 1.
    p=23: 23. Size 1.
    p=29: 29. Size 1.
    p=31: 31. Size 1.
    p=37: 37. Size 1.
    
    Total: 3 + 2 + 2 + 9 = 16.
    """
    # Construct the set
    # p=2: exponents 1, 2, 4 -> numbers 2^1, 2^2, 2^4 = 2, 4, 16
    # p=3: exponents 1, 2 -> numbers 3^1, 3^2 = 3, 9
    # p=5: exponents 1, 2 -> numbers 5^1, 5^2 = 5, 25
    # p=7, 11, 13, 17, 19, 23, 29, 31, 37: just the prime itself
    
    result = [
        2, 4, 16,  # 2^1, 2^2, 2^4
        3, 9,      # 3^1, 3^2
        5, 25,     # 5^1, 5^2
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37
    ]
    
    # Ensure types are correct and within range
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
