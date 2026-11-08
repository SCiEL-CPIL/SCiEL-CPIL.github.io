"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

D = 11
# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # safety margin from 120s

    # Generate all weight-4 vectors in R^11
    # A weight-4 vector has exactly 4 non-zero entries, each in {-1, 1}
    # Total count: C(11,4) * 2^4 = 330 * 16 = 5280
    vectors = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    n = len(vectors)
    
    if n < 2:
        return vectors
    
    # Compute dot product matrix
    # v_i . v_j = sum of products
    # For weight-4 vectors, dot product is sum over common positions of sign_i * sign_j
    # Dot product range: -4 to 4
    # Conflict condition: dot product > 2 (i.e., dot product >= 3)
    # Since all values are integers, dot product > 2 means dot product >= 3
    
    # Compute Gram matrix
    G = vectors @ vectors.T
    
    # Build adjacency: conflict if G[i,j] > 2
    # For efficiency, use vectorized operations
    conflict = G > 2
    np.fill_diagonal(conflict, False)
    
    # Greedy independent set: repeatedly pick vertex with minimum degree in remaining graph
    # Use a priority-based approach
    
    in_set = np.zeros(n, dtype=bool)
    remaining = np.ones(n, dtype=bool)
    
    # Track degrees
    degrees = conflict.sum(axis=1)
    
    while remaining.any():
        if time.time() > deadline:
            break
        
        # Find vertex with minimum degree among remaining
        # Use argmin with mask
        masked_degrees = np.where(remaining, degrees, np.inf)
        idx = np.argmin(masked_degrees)
        
        if not remaining[idx]:
            break
        
        # Add to independent set
        in_set[idx] = True
        remaining[idx] = False
        
        # Remove neighbors from remaining
        neighbors = conflict[idx] & remaining
        remaining[neighbors] = False
        
        # Update degrees for remaining vertices
        # This is expensive, but for ~5000 vertices and sparse conflicts, should be OK
        # Actually, let's just recompute degrees periodically or use a simpler approach
        # For now, recompute degrees for remaining vertices
    
    result = vectors[in_set]
    return result
# EVOLVE-BLOCK-END
