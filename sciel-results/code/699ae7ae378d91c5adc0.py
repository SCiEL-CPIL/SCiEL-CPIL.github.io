import numpy as np
import itertools

D = 11

def construct():
    # The optimal strategy for maximizing the number of integer vectors
    # with min_dist >= max_norm is to use a set of vectors with equal norm
    # such that the minimum pairwise distance squared is exactly equal to the norm squared.
    
    # For norm squared = 4, the vectors are:
    # 1. +/- 2 * e_i (22 vectors)
    # 2. +/- e_i +/- e_j +/- e_k +/- e_l (4 non-zero entries of 1)
    #    Count: C(11, 4) * 2^4 = 330 * 16 = 5280 vectors
    #
    # Total: 5302 vectors.
    #
    # Verification:
    # Max norm squared = 4.
    # Min distance squared between any two distinct vectors in this set is 4.
    # - Dist between 2e_i and 2e_j (i!=j) is 8.
    # - Dist between 2e_i and -2e_i is 16.
    # - Dist between 2e_i and (e_i+e_j+e_k+e_l) is 4 (if i is in the set of 4) or 8 (if not).
    # - Dist between two 4-one vectors:
    #   - If they share 3 indices: diff is 2e_k, norm sq 4.
    #   - If they share 2 indices: diff is 2e_k+2e_l, norm sq 8.
    #   - If they share 1 index: diff is 2e_k+2e_l+2e_m, norm sq 12.
    #   - If they share 0 indices: diff is e_k+e_l+e_m+e_n-e_o-e_p-e_q-e_r, norm sq 8.
    #   - Note: Can they share 4 indices? Yes, if they are the same vector (not distinct) or same support but different signs.
    #     If same support, e.g., e1+e2+e3+e4 and e1+e2+e3-e4. Diff 2e4. Norm sq 4.
    #
    # Thus, all pairs satisfy dist_sq >= 4 = max_norm_sq.
    
    vectors = []
    
    # Type 1: +/- 2 * e_i
    for i in range(D):
        v1 = np.zeros(D, dtype=np.int64)
        v1[i] = 2
        vectors.append(v1)
        
        v2 = np.zeros(D, dtype=np.int64)
        v2[i] = -2
        vectors.append(v2)
        
    # Type 2: 4 ones with signs
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
