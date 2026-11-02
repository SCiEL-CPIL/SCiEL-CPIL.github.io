import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Generate all weight-2 vectors: C(11,2)*4 = 220
    w2_vectors = []
    w2_indices = []  # (i, j) pairs
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                w2_vectors.append((i, j, si, sj))
    
    # Generate all weight-4 vectors: C(11,4)*16 = 330*16 = 5280
    w4_vectors = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            w4_vectors.append((combo, signs))
    
    n_w2 = len(w2_vectors)  # 220
    n_w4 = len(w4_vectors)  # 5280
    total = n_w2 + n_w4
    
    # For each weight-4 vector, compute its conflict set with all weight-2 vectors
    # Two vectors conflict if their distance squared < 4 (i.e., <= 3)
    # ||v2 - v4||^2 = ||v2||^2 + ||v4||^2 - 2*v2.v4 = 2 + 4 - 2*v2.v4 = 6 - 2*v2.v4
    # Conflict if 6 - 2*v2.v4 < 4 => 2 < 2*v2.v4 => 1 < v2.v4 => v2.v4 >= 2
    # v2.v4 >= 2 means the dot product is at least 2
    
    # Precompute conflict masks: for each w4, which w2's it conflicts with
    # This is a n_w4 x n_w2 boolean matrix
    w2_arr = np.zeros((n_w2, D), dtype=np.int8)
    for idx, (i, j, si, sj) in enumerate(w2_vectors):
        w2_arr[idx, i] = si
        w2_arr[idx, j] = sj
    
    # For each w4, compute dot products with all w2's
    # w4 as array
    w4_arr = np.zeros((n_w4, D), dtype=np.int8)
    for idx, (combo, signs) in enumerate(w4_vectors):
        for pos, (coord, sign) in enumerate(zip(combo, signs)):
            w4_arr[idx, coord] = sign
    
    # Dot products: n_w4 x n_w2
    # dot[i, j] = w4_arr[i] . w2_arr[j]
    # Conflict if dot >= 2
    dot_products = w4_arr.astype(np.int16) @ w2_arr.astype(np.int16).T
    conflict_w4_w2 = (dot_products >= 2)  # n_w4 x n_w2 boolean
    
    # Also need conflicts among w4 vectors themselves
    # ||v4a - v4b||^2 = 4 + 4 - 2*v4a.v4b = 8 - 2*v4a.v4b
    # Conflict if 8 - 2*v4a.v4b < 4 => 4 < 2*v4a.v4b => 2 < v4a.v4b => v4a.v4b >= 3
    # But max dot product of two weight-4 vectors is 4 (if identical), so conflict if dot >= 3
    # Wait, they can't be identical in a set. Let me reconsider.
    # Actually for the independent set, we just need pairwise non-conflict.
    # Two w4 vectors conflict if ||va - vb||^2 < 4, i.e., 8 - 2*dot < 4, i.e., dot > 2, i.e., dot >= 3.
    
    # This is a large matrix (5280 x 5280), let's compute it
    w4_dot = w4_arr.astype(np.int16) @ w4_arr.astype(np.int16).T
    conflict_w4_w4 = (w4_dot >= 3)
    np.fill_diagonal(conflict_w4_w4, False)
    
    # Now we want to find a large independent set in the combined graph.
    # Strategy: Use a greedy approach that prioritizes w2 vectors (they have fewer conflicts)
    # and uses a weighted scoring.
    
    # Alternative approach: Start with all w2 vectors, then greedily add w4 vectors
    # that don't conflict with the selected set.
    
    best_set = []
    best_score = 0
    
    # Approach 1: Start with a subset of w2, then add compatible w4
    # The w2 vectors among themselves: ||v2a - v2b||^2 = 2+2-2*dot = 4-2*dot
    # Conflict if 4-2*dot < 4 => dot > 0 => dot >= 1
    # So two w2 vectors conflict if their dot product is >= 1.
    
    # Let's try multiple random orderings and greedy selection
    rng = np.random.default_rng(42)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Random permutation of all vectors
        all_indices = np.arange(total)
        rng.shuffle(all_indices)
        
        selected = set()
        selected_w2 = set()
        selected_w4 = set()
        
        for idx in all_indices:
            if time.time() > deadline:
                break
            
            if idx < n_w2:
                # It's a w2 vector
                # Check conflicts with selected w2's
                v2 = w2_arr[idx]
                conflicts = False
                for sel_idx in selected_w2:
                    if np.dot(v2, w2_arr[sel_idx]) >= 1:
                        conflicts = True
                        break
                if not conflicts:
                    # Check conflicts with selected w4's
                    for sel_idx in selected_w4:
                        if conflict_w4_w2[sel_idx, idx]:
                            conflicts = True
                            break
                if not conflicts:
                    selected_w2.add(idx)
            else:
                # It's a w4 vector
                w4_idx = idx - n_w2
                conflicts = False
                # Check conflicts with selected w2's
                for sel_idx in selected_w2:
                    if conflict_w4_w2[w4_idx, sel_idx]:
                        conflicts = True
                        break
                if not conflicts:
                    # Check conflicts with selected w4's
                    for sel_idx in selected_w4:
                        if conflict_w4_w4[w4_idx, sel_idx]:
                            conflicts = True
                            break
                if not conflicts:
                    selected_w4.add(w4_idx)
        
        current_score = len(selected_w2) * 2 + len(selected_w4)
        if current_score > best_score:
            best_score = current_score
            best_set = list(selected_w2) + [x + n_w2 for x in selected_w4]
    
    # Build the result array
    if not best_set:
        # Fallback: just return a few non-conflicting vectors
        result = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64)
        return result
    
    result = np.zeros((len(best_set), D), dtype=np.int64)
    for i, idx in enumerate(best_set):
        if idx < n_w2:
            combo, si, sj = w2_vectors[idx][0], w2_vectors[idx][2], w2_vectors[idx][3]
            i_coord, j_coord = w2_vectors[idx][0], w2_vectors[idx][1]
            result[i, i_coord] = si
            result[i, j_coord] = sj
        else:
            w4_idx = idx - n_w2
            combo, signs = w4_vectors[w4_idx]
            for pos, (coord, sign) in enumerate(zip(combo, signs)):
                result[i, coord] = sign
    
    return result
