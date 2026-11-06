import time
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # Precompute coprime relationships
    # coprime_sets[a] is a set of b in 1..511 such that gcd(a,b)==1
    coprime_sets = [set() for _ in range(512)]
    for a in range(1, 512):
        for b in range(1, 512):
            if gcd(a, b) == 1:
                coprime_sets[a].add(b)
    
    # Greedy Initialization
    S = set()
    for a in range(511, 0, -1):
        if S.issubset(coprime_sets[a]):
            S.add(a)
            
    current_score = sum(1.0 / (512 - a) for a in S)
    
    # Local Search (Hill Climbing)
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try swaps: replace a_out with a_in where a_in > a_out
        # Iterate over a_out in S
        for a_out in list(S):
            # Find best a_in > a_out
            best_a_in = None
            best_gain = 0
            # We only care about a_in > a_out to increase score
            # Check candidates from 511 down to a_out + 1
            for a_in in range(511, a_out, -1):
                if a_in in S:
                    continue
                # Check if a_in is coprime with S \ {a_out}
                # S \ {a_out} is a set. coprime_sets[a_in] is a set.
                if (S - {a_out}).issubset(coprime_sets[a_in]):
                    gain = 1.0 / (512 - a_in) - 1.0 / (512 - a_out)
                    if gain > best_gain:
                        best_gain = gain
                        best_a_in = a_in
            
            if best_a_in is not None:
                S.remove(a_out)
                S.add(best_a_in)
                current_score += best_gain
                improved = True
                # Break to restart loop with new set, as other swaps might be invalidated
                break
        
        # Try adds: add a_in if coprime with all in S
        if not improved:
            best_a_in = None
            best_gain = 0
            for a_in in range(511, 0, -1):
                if a_in in S:
                    continue
                if S.issubset(coprime_sets[a_in]):
                    gain = 1.0 / (512 - a_in)
                    if gain > best_gain:
                        best_gain = gain
                        best_a_in = a_in
            
            if best_a_in is not None:
                S.add(best_a_in)
                current_score += best_gain
                improved = True

    return list(S)
