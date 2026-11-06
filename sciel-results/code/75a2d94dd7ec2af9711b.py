"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    """
    Finds the subset S of [1, 512] with maximum cardinality such that 
    lcm(a, b) <= 512 for all a, b in S.
    
    The problem is equivalent to finding an integer L <= 512 that maximizes 
    the number of divisors d(L). The optimal set S is the set of all divisors 
    of such an L. This can be viewed as finding the maximum weight closure 
    in the divisor lattice, which reduces to selecting the principal ideal 
    with the maximum weight (cardinality).
    """
    max_divisors = 0
    best_L = 1
    best_divisors = []
    
    # Iterate through all possible LCM values L <= 512
    for L in range(1, 513):
        # Count divisors of L
        divs = []
        for i in range(1, int(L**0.5) + 1):
            if L % i == 0:
                divs.append(i)
                if i != L // i:
                    divs.append(L // i)
        
        if len(divs) > max_divisors:
            max_divisors = len(divs)
            best_L = L
            best_divisors = divs
            
    # Return the sorted list of divisors of the best L
    return sorted(best_divisors)
# EVOLVE-BLOCK-END
