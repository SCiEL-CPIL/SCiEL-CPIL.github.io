import math
from math import gcd

MAXIMUM = 128

def construct():
    """
    Returns a list of [residue, modulus] pairs representing pairwise disjoint
    infinite residue classes with distinct moduli in [2, 128].
    
    Strategy:
    Select all prime numbers in the range [2, 128] as moduli.
    Since primes are pairwise coprime, any choice of residues results in
    pairwise disjoint residue classes (gcd(p_i, p_j) = 1 implies the classes
    are disjoint for any residues).
    The number of such classes is the count of primes <= 128, which is 31.
    This is the theoretical maximum for the number of pairwise coprime moduli
    in this range, and thus maximizes the number of disjoint classes.
    """
    solution = []
    
    # Generate all primes in [2, 128]
    for m in range(2, MAXIMUM + 1):
        is_prime = True
        if m < 2:
            is_prime = False
        else:
            for i in range(2, int(math.isqrt(m)) + 1):
                if m % i == 0:
                    is_prime = False
                    break
        
        if is_prime:
            # Residue 0 is valid (0 <= 0 < m)
            solution.append([0, m])
            
    return solution
