"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    # P = lcm(1, ..., 12)
    P = 27720
    n = 12
    max_L_limit = 150  # Safe upper bound based on problem statement (max 144)
    
    best_m = 0
    max_L_found = 0
    
    # Precompute indices sorted by "tightness"? 
    # The tightness depends on L and r.
    # But we can sort inside the function.
    
    # Helper to check if a matching exists for a given r and L
    def can_match(r, L):
        # Candidates for each i in 1..12
        # j is in 1..L such that (r+j) % i == 0
        cands = []
        for i in range(1, 13):
            # We need j such that r + j = k * i for some integer k.
            # j = k * i - r
            # j >= 1 => k * i >= r + 1
            # j <= L => k * i <= r + L
            
            # Smallest multiple of i > r
            # first_mult = ((r // i) + 1) * i
            # Then j = first_mult - r
            # Subsequent j: first_j + i, first_j + 2i, ...
            
            # Simpler:
            # rem = (-r) % i
            # if rem == 0, first_j = i (since j must be >= 1, and r is multiple, next is r+i)
            # wait, if r=12, i=12. r+1..r+L. 12+12=24. j=12.
            # (-12) % 12 = 0. first_j = 12?
            # Let's check: j=12. 12+12=24. 24%12=0. Correct.
            # If r=11, i=12. (-11)%12 = 1. first_j=1. 11+1=12. 12%12=0. Correct.
            
            rem = (-r) % i
            if rem == 0:
                first_j = i
            else:
                first_j = rem
                
            # Generate list
            lst = []
            # Optimization: if L is small, just loop
            for j in range(first_j, L + 1, i):
                lst.append(j)
            
            if not lst:
                return False
            cands.append(lst)
        
        # Sort by number of candidates (most constrained first)
        cands.sort(key=len)
        
        # Backtracking
        used_mask = 0
        # Use recursion with bitmask
        # Max L is 150, so bitmask needs 150 bits. Python int is fine.
        
        def backtrack(idx):
            nonlocal used_mask
            if idx == 12:
                return True
            for j in cands[idx]:
                bit = 1 << j
                if not (used_mask & bit):
                    used_mask |= bit
                    if backtrack(idx + 1):
                        return True
                    used_mask ^= bit
            return False
        
        return backtrack(0)

    start_time = time.time()
    deadline = start_time + 50.0  # Keep safety margin
    
    for r in range(P):
        # Check time
        if time.time() > deadline:
            break
            
        # Find minimal L for this r
        # We know L is at least 1.
        # We can iterate L from 1 to max_L_limit
        found_L = None
        for L in range(1, max_L_limit + 1):
            if can_match(r, L):
                found_L = L
                break
        
        if found_L is not None:
            if found_L > max_L_found:
                max_L_found = found_L
                best_m = r
                
    return int(best_m)
# EVOLVE-BLOCK-END
