"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Finds the subset S of [1, 512] with maximum cardinality such that 
    lcm(a, b) <= 512 for all a, b in S.
    
    Strategy: The optimal set is the set of divisors of the integer L <= 512 
    that has the maximum number of divisors. This is because any valid set 
    S is a subset of the divisors of L = lcm(S), and if L <= 512, the set of 
    all divisors of L is a valid set.
    """
    max_divisors = 0
    best_divisors = []
    
    # Iterate through all possible L <= 512
    for L in range(1, 513):
        # Calculate divisors of L
        divs = []
        sqrt_L = int(L**0.5)
        for i in range(1, sqrt_L + 1):
            if L % i == 0:
                divs.append(i)
                if i != L // i:
                    divs.append(L // i)
        
        if len(divs) > max_divisors:
            max_divisors = len(divs)
            best_divisors = divs
            
    # Return the sorted list of divisors of the best L
    return sorted(best_divisors)
# EVOLVE-BLOCK-END
