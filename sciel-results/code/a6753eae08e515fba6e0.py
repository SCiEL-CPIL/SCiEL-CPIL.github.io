import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Generate candidate vectors
    # Norm 2: 2 non-zero entries (+/- 1)
    # Count: C(11, 2) * 2^2 = 55 * 4 = 220
    # Norm 4: 4 non-zero entries (+/- 1)
    # Count: C(11, 4) * 2^4 = 330 * 16 = 5280
    # Total N = 5500
    
    vectors = []
    norms = []
    
    indices = range(D)
    
    # Norm 2 vectors
    for combo in itertools.combinations(indices, 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, val in zip(combo, signs):
                v[idx] = val
            vectors.append(v)
            norms.append(2)
            
    # Norm 4 vectors
    for combo in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(combo, signs):
                v[idx] = val
            vectors.append(v)
            norms.append(4)
            
    vectors = np.array(vectors, dtype=np.int64)
    norms = np.array(norms, dtype=np.int64)
    N = len(vectors)
    
    if time.time() > deadline:
        return vectors[:100]
        
    # Compute dot product matrix
    M = vectors @ vectors.T
    
    # Determine conflict matrix
    # Constraint: For any pair i, j in the selected set S,
    # ||v_i - v_j||^2 >= max(norm^2 in S)
    # We assume max norm^2 is 4 (since we include norm 4 vectors).
    # If the final set has no norm 4 vectors, max norm^2 is 2.
    # The constraints for max norm^2 = 4 are stricter:
    # Norm 4, Norm 4: dot <= 2
    # Norm 4, Norm 2: dot <= 1
    # Norm 2, Norm 2: dot <= 0
    # If we satisfy these, we are valid for R_max=4.
    # If the set ends up having only Norm 2 vectors, they satisfy dot <= 0, which is stricter than dot <= 1 (required for R_max=2).
    # So satisfying the R_max=4 constraints is sufficient for validity.
    
    # Conflict if:
    # (norm_i == 4 and norm_j == 4 and M[i,j] >= 3)
    # (norm_i == 4 and norm_j == 2 and M[i,j] >= 2)  (symmetric)
    # (norm_i == 2 and norm_j == 2 and M[i,j] >= 1)
    
    # We can compute this efficiently.
    # Create masks for norm 2 and norm 4
    is_n2 = (norms == 2)
    is_n4 = (norms == 4)
    
    # Initialize conflict matrix to False
    C = np.zeros((N, N), dtype=bool)
    
    # Norm 4 - Norm 4 conflicts
    # M[i,j] >= 3
    # Only check where both are norm 4
    mask_44 = is_n4[:, None] & is_n4[None, :]
    C |= (M >= 3) & mask_44
    
    # Norm 4 - Norm 2 conflicts
    # M[i,j] >= 2
    mask_42 = is_n4[:, None] & is_n2[None, :]
    C |= (M >= 2) & mask_42
    # Symmetric part
    mask_24 = is_n2[:, None] & is_n4[None, :]
    C |= (M >= 2) & mask_24
    
    # Norm 2 - Norm 2 conflicts
    # M[i,j] >= 1
    mask_22 = is_n2[:, None] & is_n2[None, :]
    C |= (M >= 1) & mask_22
    
    # Ensure diagonal is False (no self-conflict)
    np.fill_diagonal(C, False)
    
    # Greedy Independent Set with Randomized Restarts
    best_count = 0
    best_selected = []
    
    num_restarts = 50
    rng = np.random.default_rng(42)
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Shuffle indices
        perm = rng.permutation(N)
        
        available = np.ones(N, dtype=bool)
        selected = []
        
        # Iterate through shuffled indices
        for idx in perm:
            if not available[idx]:
                continue
                
            # Select idx
            selected.append(idx)
            
            # Update available: remove idx and all its neighbors
            # available = available & ~C[idx]
            # We need to set available[idx] to False as well, but C[idx,idx] is False, 
            # so ~C[idx] is True at idx. We must explicitly clear it or include it in C.
            # Let's just do:
            available[idx] = False
            neighbors = C[idx]
            available = available & ~neighbors
            
        count = len(selected)
        if count > best_count:
            best_count = count
            best_selected = selected
            
    if best_count == 0:
        # Fallback
        return vectors[:10]
        
    result = vectors[best_selected]
    return result
