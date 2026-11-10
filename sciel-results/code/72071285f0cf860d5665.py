"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The optimal strategy for maximizing the number of adjacent coprime divisor pairs
    # under the constraint n <= 10^18 is to use the largest possible primorial.
    # This maximizes the number of distinct prime factors, which increases the
    # diversity of the divisor set and the likelihood of coprime adjacencies.
    
    # The first 15 primes are:
    # 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    # Their product is 614,889,782,588,491,410
    # The next prime is 53, and 614,889,782,588,491,410 * 53 > 10^18
    
    return 614889782588491410
# EVOLVE-BLOCK-END
