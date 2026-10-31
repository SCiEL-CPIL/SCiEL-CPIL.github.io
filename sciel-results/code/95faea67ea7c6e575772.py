import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    # Generate all vectors with exactly 4 non-zero entries, each +/- 1
    # Number of such vectors: C(11,4) * 2^4 = 330 * 16 = 5280
    vectors = []
    indices = range(D)
    for combo in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(v)
            
    V = np.array(vectors, dtype=np.int64)
    n = len(V)
    
    # Compute dot products
    # To save memory/time, we can compute in blocks or use sparse logic if needed.
    # 5280 x 5280 matrix of int64 is 5280*5280*8 bytes ~ 224 MB. This is acceptable.
    dots = V @ V.T
    
    # Conflict: dot > 2
    # We want a subset where dots[i][j] <= 2 for all i != j in subset.
    # Diagonal is 4, which is fine.
    
    # Greedy Independent Set
    # Sort by degree (number of conflicts)
    # Conflict if dot > 2 and i != j
    conflict_mask = (dots > 2)
    np.fill_diagonal(conflict_mask, False)
    degrees = np.sum(conflict_mask, axis=1)
    
    # Order by degree ascending
    order = np.argsort(degrees)
    
    selected = []
    selected_indices = np.zeros(n, dtype=bool)
    
    # Greedy pass
    for idx in order:
        if time.time() > deadline - 10:
            break
        # Check if idx conflicts with any selected
        # selected_indices is a boolean mask
        # conflict_mask[idx] & selected_indices
        if not np.any(conflict_mask[idx] & selected_indices):
            selected.append(idx)
            selected_indices[idx] = True
            
    current_best = V[selected]
    
    # Try to improve with local search (if time permits)
    # Simple hill climbing: try to add a removed vector if we remove one conflicting vector?
    # Or just return the greedy result.
    
    return current_best
