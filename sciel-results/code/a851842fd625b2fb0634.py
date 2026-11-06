import time
from math import gcd

def construct():
    deadline = time.time() + 80.0
    MAX_M = 128
    
    best = None
    best_count = 0
    
    # Analytical construction:
    # For a given K, pick N multiples of K in [2, 128].
    # Assign distinct residues 0..N-1.
    # Valid if N <= K.
    # Maximize N.
    
    for K in range(2, MAX_M + 1):
        if time.time() > deadline:
            break
        
        # Get multiples of K in [2, 128]
        moduli = []
        m = K
        while m <= MAX_M:
            if m >= 2:
                moduli.append(m)
            m += K
        
        n_avail = len(moduli)
        # We can pick at most K moduli
        n_pick = min(n_avail, K)
        
        if n_pick <= best_count:
            continue
        
        # Pick the first n_pick moduli
        selected_moduli = moduli[:n_pick]
        
        # Assign residues 0, 1, ..., n_pick-1
        # Check validity (should be valid by construction)
        valid = True
        assignment = []
        for i, m in enumerate(selected_moduli):
            r = i
            if r >= m:
                r = r % m
            assignment.append([r, m])
        
        # Verify pairwise disjointness
        for i in range(len(assignment)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(assignment)):
                r1, m1 = assignment[i]
                r2, m2 = assignment[j]
                g = gcd(m1, m2)
                if r1 % g == r2 % g:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and len(assignment) > best_count:
            best_count = len(assignment)
            best = assignment[:]
            
    if best is None:
        # Fallback: single class
        best = [[0, 2]]
        
    return best
