"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import isqrt

def construct():
    n = 256
    # Precompute divisors for all possible products b*c where b,c in 2..256
    # Max product is 256*255 = 65280
    max_prod = n * (n - 1)
    divisors = [[] for _ in range(max_prod + 1)]
    for d in range(2, n + 1):
        for multiple in range(d, max_prod + 1, d):
            divisors[multiple].append(d)
    
    S = set(range(2, n + 1))
    
    # Phase 1: Enforce Antichain Property
    # If a | b and a != b, remove a (the smaller one).
    # Smaller numbers divide more, so they are more restrictive.
    changed = True
    while changed:
        changed = False
        S_list = sorted(S)
        to_remove = set()
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                if b % a == 0:
                    to_remove.add(a)
                    changed = True
                    break
        if to_remove:
            S -= to_remove
            
    # Phase 2: Remove remaining conflicts
    while True:
        degrees = {x: 0 for x in S}
        has_conflict = False
        S_list = list(S)
        
        # Iterate over all pairs (b, c)
        for i in range(len(S_list)):
            b = S_list[i]
            for j in range(i + 1, len(S_list)):
                c = S_list[j]
                prod = b * c
                # Check divisors of prod
                for a in divisors[prod]:
                    if a in S and a != b and a != c:
                        has_conflict = True
                        degrees[a] += 1
                        degrees[b] += 1
                        degrees[c] += 1
        
        if not has_conflict:
            break
        
        # Find element with max degree, tie-break by smallest value
        # Smaller numbers tend to divide more products, so they are "more restrictive".
        # Removing them might be more beneficial to clear conflicts.
        max_deg = -1
        best_candidate = None
        for x in S:
            d = degrees[x]
            if d > max_deg:
                max_deg = d
                best_candidate = x
            elif d == max_deg:
                if best_candidate is None or x < best_candidate:
                    best_candidate = x
        
        S.remove(best_candidate)
        
    return list(S)
# EVOLVE-BLOCK-END
