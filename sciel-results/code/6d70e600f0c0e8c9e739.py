"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'n': 256}

def construct():
    start_time = time.time()
    n = P['n']
    
    # Generate primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Start with primes
    S = set(primes)
    
    # Try to add composites
    # For each composite c, check if it can be added
    # Conflict if exists distinct a, b in S \ {c} such that a | b*c or b | a*c or c | a*b
    # Note: if a | b*c, since a, b, c distinct.
    
    # To speed up, we can check conflicts incrementally
    # But n=256 is small.
    
    composites = [i for i in range(2, n + 1) if not is_prime[i]]
    
    # Sort composites by some heuristic? Maybe smallest first?
    for c in composites:
        if c in S:
            continue
            
        # Check if c can be added to S
        valid = True
        # We need to check all pairs (a, b) in S
        # Condition: for all distinct a, b, c in S U {c}, a does not divide b*c.
        # This is O(|S|^2). |S| ~ 55. 55^2 = 3025.
        # 200 composites * 3025 = 600,000 ops. Fast.
        
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                # Check a | b*c
                if (b * c) % a == 0:
                    valid = False
                    break
                # Check b | a*c
                if (a * c) % b == 0:
                    valid = False
                    break
                # Check c | a*b
                if (a * b) % c == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.add(c)
            
        if time.time() - start_time > 50:
            break
            
    return list(S)
# EVOLVE-BLOCK-END
