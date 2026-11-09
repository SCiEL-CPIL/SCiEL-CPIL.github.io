import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Construct the 252-vector base configuration (2E8 + 2A3)
    vectors = []
    
    # 2E8 part in first 8 dimensions
    # Roots of E8 scaled by 2
    # Type 1: (±2, ±2, 0, ...) with even number of minuses in the unscaled version?
    # Unscaled roots: (±1, ±1, 0...) with even minuses.
    # Scaled: (±2, ±2, 0...)
    # Let's generate all (±2, ±2, 0...) in Z^8
    # There are C(8,2) * 4 = 112 such vectors.
    # But only those with even number of minuses in the underlying (±1,±1) are in E8?
    # Wait, the condition for E8 roots of type (±1,±1,0..) is even number of minus signs.
    # (1,1) -> 0 minuses. OK.
    # (1,-1) -> 1 minus. No.
    # (-1,1) -> 1 minus. No.
    # (-1,-1) -> 2 minuses. OK.
    # So for each pair of indices, we have (2,2) and (-2,-2).
    # So 28 * 2 = 56 vectors?
    # Wait, I counted 240 roots for E8.
    # 56 are of type (±1,±1,0..).
    # The other 184 are of type (±1/2, ..., ±1/2).
    # When scaled by 2, these become (±1, ..., ±1) with 8 non-zero entries.
    # Number of such vectors: 2^8 / 2 = 128?
    # Condition for E8: even number of minuses.
    # So 128 vectors of type (±1, ±1, ..., ±1).
    # Total 56 + 128 = 184?
    # My count of 240 must be wrong or my generation is incomplete.
    # Let's re-verify E8 roots.
    # E8 has 240 roots.
    # 56 are (±1, ±1, 0, 0, 0, 0, 0, 0) with even minuses.
    # 184 are (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even minuses.
    # Scaled by 2:
    # 56 become (±2, ±2, 0, ...) with even minuses in the ±1 version.
    # 184 become (±1, ±1, ..., ±1) with even minuses.
    # Total 240.
    
    # Generate 56 vectors of type (±2, ±2, 0...)
    for i in range(8):
        for j in range(i+1, 8):
            # (2, 2) and (-2, -2)
            v1 = [0]*8
            v1[i] = 2; v1[j] = 2
            vectors.append(v1 + [0,0,0])
            
            v2 = [0]*8
            v2[i] = -2; v2[j] = -2
            vectors.append(v2 + [0,0,0])
            
    # Generate 184 vectors of type (±1, ..., ±1)
    # Iterate over all 2^8 sign combinations
    for mask in range(256):
        v = [0]*8
        minuses = 0
        for k in range(8):
            if mask & (1 << k):
                v[k] = -1
                minuses += 1
            else:
                v[k] = 1
        if minuses % 2 == 0:
            vectors.append(v + [0,0,0])
            
    # 2A3 part in last 3 dimensions
    # Roots of A3 scaled by 2?
    # A3 roots are e_i - e_j. Norm sqrt(2).
    # Scaled by 2: 2(e_i - e_j). Norm sqrt(8).
    # In Z^3, these are (2, -2, 0), (2, 0, -2), (0, 2, -2) and negatives.
    # Total 6 * 2 = 12 vectors.
    
    for i in range(3):
        for j in range(i+1, 3):
            v1 = [0,0,0]
            v1[i] = 2; v1[j] = -2
            vectors.append([0]*8 + v1)
            
            v2 = [0,0,0]
            v2[i] = -2; v2[j] = 2
            vectors.append([0]*8 + v2)

    V = np.array(vectors, dtype=np.int64)
    
    # Local Search
    # We try to add more vectors.
    # Candidates: integer vectors with small norm.
    # We restrict coordinates to {-2, -1, 0, 1, 2} to keep search space manageable.
    # Max norm squared 8 (to keep R^2=8).
    
    # Precompute squared norms of existing V
    V_norms_sq = np.sum(V**2, axis=1)
    R_sq = np.max(V_norms_sq) # Should be 8
    
    added_count = 0
    # Generate candidates
    # To avoid duplicates, we can keep a set of tuples.
    V_set = {tuple(v) for v in V}
    
    # Iterate over possible candidates
    # We'll do a few passes or a single pass over a large set of candidates.
    # Given 11D, {-2..2}^11 is 5^11 = 48M candidates. Too many.
    # We can be smarter.
    # Only consider vectors with norm squared <= 8.
    
    # List of possible coordinate patterns for norm squared <= 8
    # We can generate all vectors in Z^11 with sum of squares <= 8.
    # This is a finite set.
    # Number of such vectors is small.
    
    candidates = []
    # Recursively generate or use itertools
    # Max coord value 2 (since 2^2=4, 2*4=8)
    # Actually, if one coord is 3, 3^2=9 > 8. So coords in {-2, -1, 0, 1, 2}.
    
    # We can just iterate over all combinations of small coords?
    # 5^11 is too big.
    # But we only need those with norm <= 8.
    
    # Let's generate them.
    def gen_candidates(d, max_norm_sq):
        if d == 0:
            yield []
            return
        for val in range(-2, 3):
            if val*val > max_norm_sq:
                continue
            for rest in gen_candidates(d-1, max_norm_sq - val*val):
                yield [val] + rest
                
    # This might be slow in Python if not optimized.
    # But 5^11 is the upper bound of the loop, pruned heavily.
    # Number of integer vectors in Z^11 with norm <= sqrt(8) is small.
    # Let's count roughly.
    # It's the number of points in a ball of radius 2.82 in Z^11.
    # This is manageable.
    
    try:
        for cand in gen_candidates(D, 8):
            if time.time() > deadline:
                break
            if tuple(cand) in V_set:
                continue
            c = np.array(cand, dtype=np.int64)
            c_norm_sq = np.sum(c**2)
            if c_norm_sq == 0:
                continue
            
            # Check distance to all existing vectors
            # dist_sq = ||c - v||^2 = ||c||^2 + ||v||^2 - 2 c.v
            # We need dist_sq >= R_sq (which is 8)
            # So ||c||^2 + ||v||^2 - 2 c.v >= 8
            # ||v||^2 is 8 for all v in V (base config).
            # So c_norm_sq + 8 - 2 c.v >= 8
            # c_norm_sq >= 2 c.v
            # c.v <= c_norm_sq / 2
            
            dots = V @ c
            max_dot = np.max(dots)
            if max_dot <= c_norm_sq / 2.0:
                # Valid addition
                V = np.vstack([V, c])
                V_set.add(tuple(c))
                added_count += 1
    except Exception:
        pass

    return V
