"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return the largest known subset of [1,255] containing 255
    such that gcd(a,b) > 1 for all distinct pairs a, b in the set.
    
    Strategy: 255 = 3 * 5 * 17. The smallest prime factor is 3.
    Taking all multiples of 3 in [1, 255] gives a set of size floor(255/3) = 85.
    Every pair in this set has gcd >= 3 > 1, and 255 is included.
    
    Analysis of optimality:
    Any valid set S containing 255 must consist of numbers non-coprime to 255.
    The set of all such numbers (size 127) is not a clique (e.g., 3 and 5 are coprime).
    The largest cliques in the non-coprime graph of [1, 255] are formed by multiples of a single prime.
    Multiples of 3: size 85.
    Multiples of 5: size 51.
    Multiples of 17: size 15.
    Mixing multiples of different primes requires removing elements that are coprime to the new prime,
    which results in a net loss of cardinality (e.g., adding 5 to multiples of 3 requires removing 68 elements).
    Thus, the set of multiples of 3 is optimal with size 85.
    """
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
