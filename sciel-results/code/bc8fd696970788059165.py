"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product, combinations
import random

def construct():
    # Generate candidates with squared norm = 4 (Norm = 2)
    # These are in the D11 lattice (even sum of coordinates)
    candidates = []
    
    # Type 1: One entry is +/- 2
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Type 2: Four entries are +/- 1
    # Indices
    for idx_tuple in combinations(range(D), 4):
        i, j, k, l = idx_tuple
        for signs in product((1, -1), repeat=4):
            v = np.zeros(D, dtype=np.int64)
            v[i], v[j], v[k], v[l] = signs
            candidates.append(v)
            
    # Greedy selection with random restarts to find a large set
    # Condition: min pairwise distance >= max norm (which is 2 for all)
    # So min pairwise distance >= 2
    # Squared distance >= 4
    
    best_set = []
    best_score = 0
    
    # Run multiple random greedy passes
    for _ in range(50):
        random.shuffle(candidates)
        selected = []
        for v in candidates:
            # Check if v is at distance >= 2 from all selected
            # dist^2 >= 4
            ok = True
            for w in selected:
                diff = v - w
                if np.dot(diff, diff) < 4:
                    ok = False
                    break
            if ok:
                selected.append(v)
        
        if len(selected) > best_score:
            best_score = len(selected)
            best_set = selected
            
    # If the greedy on norm 2 fails to beat a simple norm sqrt(2) set, 
    # we should consider that too. 
    # But norm 2 greedy is likely to find > 220.
    # For comparison, the norm sqrt(2) set has exactly 220 vectors.
    
    # Fallback: if somehow best_set is small, use the 220 vector set
    if best_score < 220:
         # Construct the 220 vector set
         vecs = []
         for i in range(D):
             for j in range(i+1, D):
                 for si in (1, -1):
                     for sj in (1, -1):
                         v = np.zeros(D, dtype=np.int64)
                         v[i] = si
                         v[j] = sj
                         vecs.append(v)
         return np.array(vecs, dtype=np.int64)

    return np.array(best_set, dtype=np.int64)
# EVOLVE-BLOCK-END
