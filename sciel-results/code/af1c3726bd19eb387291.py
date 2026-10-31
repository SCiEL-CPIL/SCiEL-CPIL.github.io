import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    D = 11
    # Generate all weight-4 vectors: exactly 4 non-zero entries, each ±1
    # norm^2 = 4 for all
    all_vectors = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            all_vectors.append(v)
    
    all_vectors = np.array(all_vectors, dtype=np.int8)
    n_total = len(all_vectors)
    
    # Greedy selection: pick vectors one by one, maintaining dot product <= 2 with all selected
    # For speed, use vectorized dot product checks
    
    # Strategy: randomize order for better packing, try multiple random orderings
    # First, try a deterministic approach, then random restarts if time permits
    
    best_selected = None
    best_count = 0
    
    # Precompute: we'll use a boolean mask of "available" vectors
    # Greedy: iterate through candidates, add if compatible with all selected
    
    # For efficiency, process in batches or use vectorized operations
    
    # Approach: maintain selected as a list, for each candidate check dot products
    # With vectorized numpy, checking one candidate against all selected is O(n_selected * D)
    
    # Better: maintain a "conflict count" array. For each candidate j, conflict_count[j] = 
    # number of selected vectors that conflict with j. Add j if conflict_count[j] == 0.
    # When we add vector i, we need to update conflict_count for all remaining candidates.
    
    # This is O(n^2 * D) total which for n=5280 is about 5280^2 * 11 / 2 ≈ 1.5e8 operations
    # That might be too slow in Python. Let's use numpy vectorization.
    
    # Alternative: batch processing. Process candidates in chunks.
    
    # Actually, let's use a smarter approach:
    # 1. Compute the full 5280 x 5280 dot product matrix (might be too large: 5280^2 * 8 bytes ≈ 224 MB)
    #    That's feasible!
    
    # Compute dot product matrix
    # all_vectors is (5280, 11), dot products: (5280, 5280)
    # 5280 * 5280 * 8 = ~224 MB for int32, or ~112 MB for int16
    # Let's use int8 since values range from -4 to 4
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        # Not enough time, return empty or minimal
        return np.array([], dtype=np.int64).reshape(0, D)
    
    # Compute dot product matrix
    # Using matrix multiplication
    dot_mat = all_vectors.astype(np.int16) @ all_vectors.astype(np.int16).T  # (5280, 5280) int16
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    # Now greedy selection using the precomputed dot product matrix
    # A vector i is compatible with selected set S if for all j in S: dot_mat[i,j] <= 2
    
    # Strategy: multiple random orderings, greedy select
    # For each ordering, greedily pick vectors that are compatible with all previously picked
    
    # To speed up: maintain a "blocked" bitmask or boolean array
    # When we pick vector i, all vectors j with dot_mat[i,j] > 2 become blocked
    
    n = n_total
    rng = np.random.default_rng(42)
    
    best_selected_indices = None
    best_k = 0
    
    # Try several random orderings
    max_restarts = 50
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        # Random permutation
        order = rng.permutation(n)
        
        selected = []
        # blocked: boolean array, True if vector is incompatible with some selected vector
        blocked = np.zeros(n, dtype=bool)
        
        for idx in order:
            if blocked[idx]:
                continue
            # Select this vector
            selected.append(idx)
            # Block all vectors that conflict with idx
            # Conflict: dot_mat[idx, :] > 2
            conflicts = dot_mat[idx] > 2
            blocked |= conflicts
        
        k = len(selected)
        if k > best_k:
            best_k = k
            best_selected_indices = selected
        
        # Early termination if we hit a good target
        if best_k >= 450:
            break
    
    if best_selected_indices is None:
        # Fallback: just return D11 weight-2 vectors
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result = all_vectors[best_selected_indices].astype(np.int64)
    return result
