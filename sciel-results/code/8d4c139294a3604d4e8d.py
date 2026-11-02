import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Generate all vectors with exactly 4 non-zero entries (+/- 1)
    # Number of such vectors: C(11, 4) * 2^4 = 330 * 16 = 5280
    vectors = []
    indices = range(D)
    for combo in itertools.combinations(indices, 4):
        # Generate all 16 sign combinations for the 4 positions
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(combo, signs):
                v[idx] = val
            vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    N = len(vectors)
    
    if time.time() > deadline:
        return vectors[:100] # Fallback, though generation is fast

    # Compute dot product matrix
    # M[i, j] = vectors[i] . vectors[j]
    M = vectors @ vectors.T
    
    # Greedy selection of independent set
    # Condition: M[i, j] <= 2 for all distinct i, j in set
    # Conflicts if M[i, j] >= 3
    
    mask = np.ones(N, dtype=bool)
    selected_indices = []
    
    while mask.any():
        if time.time() > deadline:
            break
            
        # Find the first available index
        idx = np.where(mask)[0][0]
        selected_indices.append(idx)
        
        # Update mask: exclude any vector j that conflicts with idx
        # Conflict if M[j, idx] >= 3
        mask = mask & (M[:, idx] < 3)
        
    result = vectors[selected_indices]
    return result
