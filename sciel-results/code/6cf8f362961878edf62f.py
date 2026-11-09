"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import lcm
    from itertools import combinations
    
    n = 128
    vertices = list(range(1, n+1))
    
    # Generate all prime powers up to n
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127]
    prime_powers = set([1])
    for p in primes:
        power = p
        while power <= n:
            prime_powers.add(power)
            power *= p
    
    # Start with prime powers
    included = set(prime_powers)
    
    # Try to add remaining vertices in order of "safety" (fewer conflicts first)
    # Compute for each vertex not in included, how many pairs in included it conflicts with
    remaining = [v for v in vertices if v not in included]
    
    # Sort by number of conflicts (ascending) - add safe ones first
    def count_conflicts(v):
        count = 0
        for a, b in combinations(included, 2):
            if lcm(a,b) == lcm(a,v) == lcm(b,v):
                count += 1
        return count
    
    remaining.sort(key=count_conflicts)
    
    for v in remaining:
        # Check if adding v is valid
        valid = True
        for a, b in combinations(included, 2):
            if lcm(a,b) == lcm(a,v) == lcm(b,v):
                valid = False
                break
        if valid:
            included.add(v)
    
    return sorted(included)
# EVOLVE-BLOCK-END
