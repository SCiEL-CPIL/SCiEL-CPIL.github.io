"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np

def construct():
    N = 128
    # Vertices are integers 1..128
    # We want to find a maximum subset S of {1..128} such that there do NOT exist
    # three distinct members a, b, c in S for which a+b, a+c, and b+c are also in S.
    
    # The current seed returns {65, 66, ..., 128} which has size 64.
    # Let's check if we can do better.
    
    # Key insight: If we pick numbers from the upper half, the sums a+b, a+c, b+c 
    # will be > 128 (since min a,b,c >= 65, so min sum = 65+66=131 > 128).
    # Thus, no pair sums are in the domain, so the condition is vacuously satisfied.
    # This gives size 64.
    
    # Can we add some numbers from the lower half?
    # If we add a small number x, we need to ensure no triple (a,b,c) with x involved
    # has all three pair sums in S.
    
    # Let's try a branch-and-bound or greedy approach with conflict detection.
    # But first, let's think about the structure.
    
    # A forbidden configuration is {a,b,c} distinct in S such that a+b, a+c, b+c all in S.
    # Note that a+b, a+c, b+c must be distinct from a,b,c? Not necessarily, but they must be in S.
    
    # Strategy: Start with the upper half {65..128} and try to add elements from {1..64}.
    # When adding an element x, check if it creates a forbidden triple with existing elements.
    
    # Actually, let's model this as a maximum independent set in a 3-uniform hypergraph.
    # But that's expensive. Let's try a greedy approach with backtracking.
    
    # Alternative: Use the fact that if S contains only numbers > 64, it's valid.
    # Can we include some numbers <= 64?
    
    # Let's try: S = {65..128} ∪ T where T ⊆ {1..64}
    # For any triple (a,b,c) in S, if all three are > 64, their sums are > 128, so safe.
    # If two are > 64 and one is x <= 64: say a,b > 64, c=x.
    #   Then a+b > 128 (not in S), a+x and b+x could be in S.
    #   But we need ALL THREE sums in S. Since a+b > 128, this triple is safe.
    # If one is > 64 and two are <= 64: say a > 64, b,c <= 64.
    #   a+b, a+c > 64, could be in S. b+c <= 128, could be in S.
    #   Need all three in S.
    # If all three are <= 64: a,b,c <= 64.
    #   a+b, a+c, b+c <= 128, could all be in S.
    
    # So the danger is:
    # 1. Three elements all in T (lower part): their pair sums must not all be in S.
    # 2. One element in upper, two in T: the two upper+lower sums and the lower+lower sum must not all be in S.
    
    # Let's try a greedy construction:
    # Start with S = set(range(65, 129))
    # Try to add elements from 64 down to 1.
    # For each candidate x, check if adding x creates a forbidden triple.
    
    S = set(range(65, 129))
    
    # Precompute for efficiency
    for x in range(64, 0, -1):
        # Check if adding x to S creates a forbidden triple
        # A forbidden triple involves x and two other elements a, b in S
        # such that x+a, x+b, a+b are all in S ∪ {x}
        
        valid = True
        for a in S:
            if a == x:
                continue
            for b in S:
                if b == x or b == a:
                    continue
                # Check triple {x, a, b}
                sa = x + a
                sb = x + b
                ab = a + b
                if sa in S and sb in S and ab in S:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.add(x)
    
    # Convert to list of ints
    result = sorted(S)
    # Ensure all are Python int
    result = [int(x) for x in result]
    return result
# EVOLVE-BLOCK-END
