"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 55  # safety margin
    n = 128
    
    # Start with the upper half: {65, ..., 128}
    base_S = set(range(65, n + 1))
    
    def is_valid_addition(x, S):
        """
        Check if adding x to S keeps S valid.
        Forbidden: distinct a,b,c in S U {x} such that a+b, a+c, b+c in S U {x}.
        Since S (base) is sum-free, any new violation must involve x.
        
        Case 1: x is a member. {x, b, c}.
        Requires b+c in S U {x}. Since b,c >= 65, b+c > 128. So b+c must be x.
        But x <= 128 and b+c >= 130. Impossible.
        
        Case 2: x is a sum. a+b = x.
        Requires a,b,c in S distinct, and a+c in S U {x}, b+c in S U {x}.
        If a+c = x, then c = x-a = b, contradiction (distinct).
        So a+c in S and b+c in S.
        
        So: Invalid if exists a,b in S, a != b, a+b=x, and exists c in S, c != a, c != b,
        such that a+c in S and b+c in S.
        """
        for a in S:
            b = x - a
            if b <= 0 or b == a or b not in S:
                continue
            # Found a pair a,b in S summing to x.
            # Check if there is a c in S (distinct from a,b) such that a+c in S and b+c in S.
            for c in S:
                if c == a or c == b:
                    continue
                if (a + c) in S and (b + c) in S:
                    return False
        return True

    best_set = None
    best_size = 0
    
    # Try multiple random restarts of greedy addition
    # The order of adding small numbers matters.
    
    # First, try the standard 1..64 order as a baseline
    S = set(base_S)
    for x in range(1, 65):
        if is_valid_addition(x, S):
            S.add(x)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S.copy()
        
    # Random restarts
    # We have plenty of time budget.
    # Each pass takes ~0.05s. 55s / 0.05s = 1100 restarts.
    
    while time.time() < deadline:
        S = set(base_S)
        candidates = list(range(1, 65))
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_addition(x, S):
                S.add(x)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    return sorted(best_set)
# EVOLVE-BLOCK-END
