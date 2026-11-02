import numpy as np
import itertools
import time

D = 11

def _generate_weight_k_vectors(k):
    """Generate all integer vectors in R^D with exactly k non-zero entries of magnitude 1."""
    vecs = []
    # Choose k indices from D
    for indices in itertools.combinations(range(D), k):
        # Assign signs: 2^k possibilities
        for signs in itertools.product([1, -1], repeat=k):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            vecs.append(tuple(v))
    return np.array(vecs, dtype=np.int64)

def _max_independent_set_bitset(conflict_matrix):
    """
    Find a large independent set using a greedy algorithm with bitsets.
    conflict_matrix: NxN boolean numpy array.
    """
    n = conflict_matrix.shape[0]
    # Convert conflict matrix to list of bitsets (using Python integers or numpy arrays of uint64)
    # For n=1320, we need about 21 uint64s.
    # Using Python integers is easier for bitwise ops if n < 10000.
    # 1320 bits fits in a Python int easily.
    
    neighbors = [0] * n
    for i in range(n):
        mask = 0
        row = conflict_matrix[i]
        # Set bits for j where conflict[i,j] is True
        # We can iterate through True indices
        indices = np.where(row)[0]
        for j in indices:
            mask |= (1 << j)
        neighbors[i] = mask
        
    available = (1 << n) - 1
    selected = []
    
    while available:
        # Find vertex with minimum degree in available subgraph
        # degree(v) = popcount(neighbors[v] & available)
        min_deg = n + 1
        best_v = -1
        
        # Iterate through available vertices
        # To speed up, we can just iterate all n and check if bit is set
        # n=1320 is small enough for a Python loop if optimized?
        # 1320 iterations * 1320 steps? No, just 1320 iterations to find min.
        # Each step is a bitwise AND and popcount.
        
        # Optimization: We can just pick the first one? No, min-degree is better.
        # Let's try a simple heuristic: pick the one with fewest neighbors.
        
        # To avoid O(N^2) per step (which is O(N^3) total), we can be smarter,
        # but for N=1320, N^2 is 1.7M. If we do this N times, it's 2B ops. Too slow.
        
        # Alternative: Randomized greedy or simpler heuristic.
        # Let's try: Pick a random available vertex?
        # Or: Pick the vertex with min degree among a random subset?
        
        # Actually, we can update degrees incrementally?
        # When we select v, we remove its neighbors.
        # The degree of remaining nodes decreases.
        
        # Let's stick to a simpler, faster heuristic:
        # 1. Start with all available.
        # 2. Pick v with min (neighbors[v] & available).bit_count()
        # 3. available &= ~(neighbors[v] | (1<<v))
        
        # To make it faster, we can limit the search for best_v to a random sample of available nodes?
        # Or just use the first available node? (Greedy)
        
        # Let's try the "First Available" strategy first, it's O(N^2) total.
        # Or "Min Degree" but only checking a subset.
        
        # Let's try:
        # Get list of available indices
        avail_indices = []
        temp = available
        while temp:
            # Get least significant bit
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            avail_indices.append(idx)
            temp ^= lsb
            
        if not avail_indices:
            break
            
        # Pick min degree from avail_indices
        # This is O(K) where K is number of available.
        # Total complexity O(N^2). 1320^2 = 1.7M. This is fine.
        
        min_deg = n + 1
        best_v = avail_indices[0]
        
        for v in avail_indices:
            deg = (neighbors[v] & available).bit_count()
            if deg < min_deg:
                min_deg = deg
                best_v = v
                
        selected.append(best_v)
        # Remove best_v and its neighbors
        available &= ~(neighbors[best_v] | (1 << best_v))
        
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    # 1. Generate Weight-2 vectors
    w2_vecs = _generate_weight_k_vectors(2)
    k2 = len(w2_vecs)
    
    # Weight-2 vectors are all mutually compatible (dist sq >= 2, max norm sq = 2)
    # So we have a valid set of size k2.
    best_vecs = w2_vecs
    best_k = k2
    
    # 2. Generate Weight-3 vectors
    w3_vecs = _generate_weight_k_vectors(3)
    k3_total = len(w3_vecs)
    
    if time.time() < deadline - 10:
        # Compute conflict matrix for Weight-3
        # Conflict if dist sq < 3 (i.e., <= 2)
        # dist sq = ||u||^2 + ||v||^2 - 2 u.v = 3 + 3 - 2 u.v = 6 - 2 u.v
        # dist sq <= 2  =>  6 - 2 u.v <= 2  =>  2 u.v >= 4  =>  u.v >= 2
        
        # So two weight-3 vectors conflict if their dot product is >= 2.
        # Dot product of two weight-3 vectors:
        # They have 3 non-zeros.
        # If they share 3 indices:
        #   Same signs: dot = 3. Conflict.
        #   2 same, 1 diff: dot = 1. No conflict (dist sq = 6-2=4 >= 3).
        #   1 same, 2 diff: dot = -1. No conflict.
        #   0 same: dot = -3. No conflict.
        # If they share 2 indices:
        #   Same signs on shared: dot = 2. Conflict.
        #   1 same, 1 diff: dot = 0. No conflict.
        #   0 same: dot = -2. No conflict.
        # If they share 1 index:
        #   Same sign: dot = 1. No conflict.
        #   Diff sign: dot = -1. No conflict.
        # If they share 0 indices:
        #   dot = 0. No conflict.
        
        # So conflict iff:
        # (Share 3 indices AND signs are identical) OR (Share 2 indices AND signs on shared are identical)
        
        # We can compute this efficiently.
        n = k3_total
        conflict = np.zeros((n, n), dtype=bool)
        
        # Vectorized dot product
        dot = w3_vecs @ w3_vecs.T
        # Conflict if dot >= 2
        conflict = dot >= 2
        np.fill_diagonal(conflict, False)
        
        # Find Max Independent Set
        try:
            selected_indices = _max_independent_set_bitset(conflict)
            k3_selected = len(selected_indices)
            
            if k3_selected > best_k:
                best_vecs = w3_vecs[selected_indices]
                best_k = k3_selected
        except Exception as e:
            pass
            
    # 3. Try Weight-4?
    # Weight-4 vectors: norm 2. Max norm 2.
    # Constraint: dist sq >= 4.
    # dot product condition: ||u-v||^2 = 4 + 4 - 2 dot = 8 - 2 dot >= 4 => 2 dot <= 4 => dot <= 2.
    # Conflict if dot > 2 (i.e., dot >= 3).
    
    if time.time() < deadline - 20:
        w4_vecs = _generate_weight_k_vectors(4)
        k4_total = len(w4_vecs)
        
        if k4_total < 10000: # Safety
            dot = w4_vecs @ w4_vecs.T
            conflict = dot > 2
            np.fill_diagonal(conflict, False)
            
            try:
                selected_indices = _max_independent_set_bitset(conflict)
                k4_selected = len(selected_indices)
                
                if k4_selected > best_k:
                    best_vecs = w4_vecs[selected_indices]
                    best_k = k4_selected
            except Exception:
                pass

    return best_vecs.astype(np.int64)
