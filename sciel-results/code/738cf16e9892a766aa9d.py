"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    # Greedy construction: Add numbers from largest to smallest
    # Start with empty set, which is valid.
    # For each candidate a, check if adding it to current set S creates any conflict.
    # Conflicts involving a:
    # 1. a | b*c for b,c in S, b!=c
    # 2. b | a*c for b,c in S, b!=c
    # 3. c | a*b for b,c in S, b!=c
    
    S = set()
    
    # Iterate from largest to smallest
    for a in range(N, 1, -1):
        if time.time() > deadline:
            break
        
        can_add = True
        
        # If S is small, checking is fast.
        # We iterate over pairs in S.
        # To avoid O(N^3) total, we rely on the fact that S might not grow too large?
        # Actually, worst case S size ~ 100. 100^2 * 250 ~ 2.5M ops. Fast.
        
        # Optimization: Convert S to list for faster iteration?
        # Set iteration is fine.
        
        # We need to check all pairs (b, c) in S with b != c.
        # We can iterate b, then c.
        
        # Early exit if not can_add
        for b in S:
            if not can_add:
                break
            for c in S:
                if b == c:
                    continue
                
                # Check 1: a divides b*c
                if (b * c) % a == 0:
                    can_add = False
                    break
                
                # Check 2: b divides a*c
                if (a * c) % b == 0:
                    can_add = False
                    break
                
                # Check 3: c divides a*b
                if (a * b) % c == 0:
                    can_add = False
                    break
        
        if can_add:
            S.add(a)
    
    # Convert to sorted list of ints
    result = sorted(list(S))
    
    # Ensure types are Python int
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
