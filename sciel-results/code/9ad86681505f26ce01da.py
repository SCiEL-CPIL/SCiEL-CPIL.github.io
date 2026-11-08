import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    # Safety margin: 10 seconds before the 150s hard limit
    deadline = start_time + 140.0
    
    # Generate all weight 4 vectors in R^11
    # There are C(11, 4) * 2^4 = 330 * 16 = 5280 such vectors.
    # Norm squared is 4 for all.
    # Condition for two weight-4 vectors v, w to be compatible:
    # ||v - w||^2 >= max(||v||^2, ||w||^2) = 4
    # ||v||^2 + ||w||^2 - 2 v.w >= 4
    # 4 + 4 - 2 v.w >= 4  =>  4 >= 2 v.w  =>  v.w <= 2
    # Since v, w are integer vectors with entries in {-1, 0, 1}, v.w is an integer.
    # So we require v.w <= 2.
    
    candidates = []
    indices = range(D)
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(comb, signs):
                v[idx] = val
            candidates.append(v)
    
    N = len(candidates)
    
    if N == 0:
        # Fallback (should not happen)
        return np.array([[[0]*D]], dtype=np.int64)
        
    V = np.array(candidates, dtype=np.int32)
    
    # Precompute dot product matrix
    # V is (N, D). V @ V.T is (N, N).
    # Using int32 to avoid overflow issues, though int16 would suffice (max dot 4).
    M = V @ V.T
    
    # Conflict mask: M[i, j] > 2 means i and j conflict.
    # We want to keep j if M[i, j] <= 2.
    # Note: M[i, i] = 4, so i conflicts with itself.
    
    best_count = 0
    best_indices = np.array([], dtype=int)
    
    # Greedy Independent Set with random restarts
    # We use a vectorized approach for the active set.
    
    while time.time() < deadline:
        # Random permutation of candidates
        perm = np.random.permutation(N)
        
        # active[j] is True if candidate j is still available (not selected and not conflicting with any selected)
        active = np.ones(N, dtype=bool)
        
        selected = []
        
        for idx in perm:
            if active[idx]:
                selected.append(idx)
                # Update active set:
                # Any candidate j is removed if it conflicts with idx.
                # Conflict condition: M[idx, j] > 2.
                # So we keep j if M[idx, j] <= 2.
                # This also removes idx itself because M[idx, idx] = 4 > 2.
                active &= (M[idx] <= 2)
                
                # Optional early termination if no active nodes left?
                # np.any(active) is O(N). Might be worth it if we get stuck early.
                # But checking every step might be slower than just letting the loop run
                # if the average number of steps is small.
                # Let's stick to the simple loop for now, as N=5280 is small.
                
        if len(selected) > best_count:
            best_count = len(selected)
            best_indices = np.array(selected, dtype=int)
            
            # Check time
            if time.time() >= deadline:
                break
                
    # If no indices found (should not happen), fallback to single vector
    if len(best_indices) == 0:
        best_indices = np.array([0], dtype=int)
        
    result = V[best_indices]
    return result.astype(np.int64)
