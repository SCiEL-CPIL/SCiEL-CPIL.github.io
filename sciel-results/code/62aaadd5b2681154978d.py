"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Finds the subset S of [1, 512] with maximum cardinality such that 
    lcm(a, b) <= 512 for all a, b in S.
    
    Insight: Any such set S is contained in the set of divisors of L = lcm(S).
    Since L <= 512, S is a subset of divisors(L). To maximize |S|, we should 
    take S = divisors(L) for the L in [1, 512] that maximizes the number of divisors d(L).
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
