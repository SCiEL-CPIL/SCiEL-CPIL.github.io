import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    D = 11
    vectors = []

    # E8 type 1: Two +/-2 in first 8 dims (norm^2 = 8)
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(tuple(v))

    if time.time() > deadline:
        return np.array(vectors, dtype=np.int64)

    # E8 type 2: All +/-1 in first 8 dims, even number of minuses (norm^2 = 8)
    for mask in range(256):
        minuses = 0
        v = [0] * D
        for i in range(8):
            if (mask >> i) & 1:
                v[i] = -1
                minuses += 1
            else:
                v[i] = 1
        if minuses % 2 == 0:
            vectors.append(tuple(v))

    if time.time() > deadline:
        return np.array(vectors, dtype=np.int64)

    # D3 type: Two +/-2 in last 3 dims (norm^2 = 8)
    for i, j in itertools.combinations(range(3), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[8 + i] = 2 * si
                v[8 + j] = 2 * sj
                vectors.append(tuple(v))

    if time.time() > deadline:
        return np.array(vectors, dtype=np.int64)

    vec_arr = np.array(vectors, dtype=np.int64)
    max_norm_sq = 8

    existing_set = set(vectors)

    # Generate additional candidates with norm^2 = 8 that are not in the base
    # These are vectors with various combinations of entries in {-2,-1,0,1,2}
    # whose sum of squares equals 8, but not already in the base set.
    
    # We'll generate candidates by considering different support patterns.
    # For efficiency, we focus on vectors with entries in {-2, -1, 0, 1, 2}
    # and sum of squares = 8.
    
    # Patterns of absolute values that sum of squares = 8:
    # (2,2,0,...): 2*4 = 8 -> already covered (E8 type1, D3 type)
    # (2,1,1,0,...): 4+1+1 = 6 -> no
    # (2,1,1,1,1,0,...): 4+1+1+1+1 = 8 -> yes (4 ones and one two)
    # (1,1,1,1,1,1,1,1,0,...): 8*1 = 8 -> already covered (E8 type2)
    # (2,2,1,1,0,...): 4+4+1+1 = 10 -> no
    # (2,1,1,1,0,...): 4+1+1+1 = 7 -> no
    # (2,1,1,0,...): 4+1+1 = 6 -> no
    # (2,0,...): 4 -> no
    # (1,1,1,1,1,1,0,...): 6 -> no
    # (1,1,1,1,0,...): 4 -> no
    # (2,2,0,...): 8 -> covered
    # (2,1,1,1,1): 8 -> new pattern
    # (1,1,1,1,1,1,1,1): 8 -> covered
    # (2,2,1,1): 10 -> no
    # (3,1,0,...): 9+1 = 10 -> no
    # (2,2,2,0,...): 12 -> no
    # (2,1,1,1,1) is the main new pattern.
    # Also (2,2,0,...) with the 2s in different positions are already covered.
    
    # Let's generate vectors with one ±2 and four ±1's (norm^2 = 4+4*1 = 8)
    # The ±2 can be in any of the 11 positions, and the four ±1's in any 4 of the remaining 10.
    
    # This is a lot: 11 * C(10,4) * 2 * 16 = 11 * 210 * 2 * 16 = 73920 candidates
    # But many will be duplicates or already in the base.
    
    candidates = []
    
    # Type A: One ±2, four ±1's (norm^2 = 8)
    for pos2 in range(D):
        for si in (1, -1):
            for ones_positions in itertools.combinations([p for p in range(D) if p != pos2], 4):
                for signs in itertools.product((1, -1), repeat=4):
                    v = [0] * D
                    v[pos2] = 2 * si
                    for k, p in enumerate(ones_positions):
                        v[p] = signs[k]
                    vt = tuple(v)
                    if vt not in existing_set:
                        candidates.append(vt)

    if time.time() > deadline - 30:
        vec_arr = np.array(vectors, dtype=np.int64)
        return vec_arr

    # Vectorized greedy addition for norm^2=8 candidates
    cand_arr = np.array(candidates, dtype=np.int64)
    n_cand = len(cand_arr)
    
    # Precompute existing norms
    vec_norms = np.sum(vec_arr**2, axis=1)
    
    batch_size = 2000
    added_in_batch = 0
    
    for start in range(0, n_cand, batch_size):
        if time.time() > deadline - 10:
            break
        end = min(start + batch_size, n_cand)
        batch = cand_arr[start:end]
        
        batch_norms = np.sum(batch**2, axis=1)
        dot_products = batch @ vec_arr.T
        dists_sq = batch_norms[:, None] + vec_norms[None, :] - 2 * dot_products
        
        min_dists = np.min(dists_sq, axis=1)
        valid_mask = min_dists >= max_norm_sq
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            new_vecs = batch[valid_indices]
            vectors.extend([tuple(v) for v in new_vecs])
            vec_arr = np.vstack([vec_arr, new_vecs])
            vec_norms = np.sum(vec_arr**2, axis=1)
            added_in_batch += len(valid_indices)
    
    # Now try to add candidates with norm^2 < 8 if time permits
    if time.time() < deadline - 20 and added_in_batch > 0:
        # Generate norm^2 = 7: one ±2, three ±1's (4+3 = 7)
        cand7 = []
        for pos2 in range(D):
            for si in (1, -1):
                for ones_positions in itertools.combinations([p for p in range(D) if p != pos2], 3):
                    for signs in itertools.product((1, -1), repeat=3):
                        v = [0] * D
                        v[pos2] = 2 * si
                        for k, p in enumerate(ones_positions):
                            v[p] = signs[k]
                        vt = tuple(v)
                        if vt not in set(vectors):
                            cand7.append(vt)
        
        if cand7 and time.time() < deadline - 15:
            cand7_arr = np.array(cand7, dtype=np.int64)
            vec_norms = np.sum(vec_arr**2, axis=1)
            
            for start in range(0, len(cand7_arr), 2000):
                if time.time() > deadline - 5:
                    break
                end = min(start + 2000, len(cand7_arr))
                batch = cand7_arr[start:end]
                batch_norms = np.sum(batch**2, axis=1)
                dot_products = batch @ vec_arr.T
                dists_sq = batch_norms[:, None] + vec_norms[None, :] - 2 * dot_products
                min_dists = np.min(dists_sq, axis=1)
                valid_mask = min_dists >= max_norm_sq
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    new_vecs = batch[valid_indices]
                    vectors.extend([tuple(v) for v in new_vecs])
                    vec_arr = np.vstack([vec_arr, new_vecs])
                    vec_norms = np.sum(vec_arr**2, axis=1)
    
    return np.array(vectors, dtype=np.int64)
