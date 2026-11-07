"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    def is_valid(S):
        n = len(S)
        max_sum = sum(S)
        # sums[k] will be a set of all possible sums of k elements
        sums = [set() for _ in range(n + 1)]
        sums[0] = {0}
        
        for x in S:
            # Iterate backwards to avoid using x multiple times in the same step
            for k in range(n - 1, -1, -1):
                if sums[k]:
                    new_sums = {s + x for s in sums[k]}
                    sums[k + 1].update(new_sums)
                    
        # Check if sums of different cardinalities are disjoint
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if sums[i] & sums[j]:
                    return False
        return True

    # Fallback: Known valid set of size 31
    fallback = [225 + i for i in range(1, 32)]
    
    # Try to find a larger set
    # We search from 60 down to 32
    # Given time constraints, we try a few heuristics per N
    for n in range(60, 31, -1):
        # Heuristic 1: Largest n numbers
        S1 = list(range(257 - n, 257))
        if is_valid(S1):
            return S1
            
        # Heuristic 2: Random samples
        # Limit random attempts to avoid timeout
        for _ in range(10):
            S2 = random.sample(range(1, 257), n)
            if is_valid(S2):
                return S2
                
    return fallback
# EVOLVE-BLOCK-END
