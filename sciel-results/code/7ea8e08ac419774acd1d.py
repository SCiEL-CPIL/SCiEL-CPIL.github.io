"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Generate all weight 4 vectors with +/- 1
    # C(11, 4) = 330 combinations of indices
    # 2^4 = 16 sign combinations
    # Total 5280 vectors
    
    candidates = []
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(indices, signs):
                v[idx] = s
            candidates.append(v)
            
    V = np.array(candidates)
    N = len(V)
    
    # Compute Gram matrix
    G = V @ V.T
    
    # Conflict if dot product >= 3
    # Norm sq is 4. Dist sq = 8 - 2 dot.
    # Dist sq >= 4 => 8 - 2 dot >= 4 => 2 dot <= 4 => dot <= 2.
    # So conflict if dot > 2.
    conflict = G >= 3
    np.fill_diagonal(conflict, False)
    
    # Greedy Independent Set
    in_set = np.zeros(N, dtype=bool)
    available = np.ones(N, dtype=bool)
    
    while available.any():
        # Calculate degree of each available vertex
        deg = (conflict & available).sum(axis=1)
        
        # Find available vertex with min degree
        deg[~available] = np.inf
        idx = np.argmin(deg)
        
        if not available[idx]:
            break
            
        in_set[idx] = True
        available[idx] = False
        available[conflict[idx]] = False
        
        # Time check
        if time.time() > deadline:
            break

    result = V[in_set]
    
    # Fallback if empty
    if len(result) == 0:
        return V[0:1]
        
    return result
# EVOLVE-BLOCK-END
