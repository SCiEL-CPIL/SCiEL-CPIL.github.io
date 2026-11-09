"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    """
    Constructs a set of nonzero integer vectors in R^11.
    Uses the roots of the D_11 lattice: vectors of the form +/-e_i +/-e_j.
    There are 2 * D * (D-1) such vectors.
    For D=11, this is 220 vectors.
    
    Verification:
    - All vectors have norm squared 2.
    - Max norm squared M = 2.
    - Min distance squared between distinct vectors:
      - If they share 0 indices: diff has 4 non-zero entries (+/-1). Norm sq 4 >= 2.
      - If they share 1 index: diff has 2 non-zero entries (+/-1). Norm sq 2 >= 2.
      - If they share 2 indices: they are either identical (excluded) or differ by sign of one component.
        e.g., e_i+e_j and e_i-e_j. Diff 2e_j. Norm sq 4 >= 2.
    - Thus, the condition is satisfied.
    """
    vectors = []
    
    # Generate all vectors of the form +/-e_i +/-e_j with i != j
    # To avoid duplicates, we can just iterate all i, j and signs.
    # Since e_i+e_j == e_j+e_i, we will generate duplicates if we are not careful,
    # but the problem asks for a set of vectors. Duplicates in the list
    # would count as separate "spheres" at the same location?
    # "minimum pairwise distance" between identical vectors is 0.
    # So we must ensure all vectors in the returned array are DISTINCT.
    
    # Strategy:
    # Iterate i < j.
    # For each pair, add:
    # v1 = e_i + e_j
    # v2 = e_i - e_j
    # v3 = -e_i + e_j
    # v4 = -e_i - e_j
    
    for i in range(D):
        for j in range(i + 1, D):
            # v1
            v1 = np.zeros(D, dtype=np.int64)
            v1[i] = 1
            v1[j] = 1
            vectors.append(v1)
            
            # v2
            v2 = np.zeros(D, dtype=np.int64)
            v2[i] = 1
            v2[j] = -1
            vectors.append(v2)
            
            # v3
            v3 = np.zeros(D, dtype=np.int64)
            v3[i] = -1
            v3[j] = 1
            vectors.append(v3)
            
            # v4
            v4 = np.zeros(D, dtype=np.int64)
            v4[i] = -1
            v4[j] = -1
            vectors.append(v4)
            
    result = np.array(vectors, dtype=np.int64)
    
    # Optional: Verification step to ensure correctness (can be removed for speed if confident)
    if len(result) > 1:
        max_norm_sq = int(np.max(np.sum(result * result, axis=1)))
        # Check min distance
        # For 220 vectors, pairwise check is 220*219/2 ~ 24k checks. Very fast.
        min_dist_sq = float('inf')
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                diff = result[i] - result[j]
                d2 = int(np.sum(diff * diff))
                if d2 < min_dist_sq:
                    min_dist_sq = d2
        
        if min_dist_sq < max_norm_sq:
            # Fallback to cross polytope if something went wrong (should not happen)
            fallback = []
            for i in range(D):
                v_pos = np.zeros(D, dtype=np.int64)
                v_pos[i] = 2
                fallback.append(v_pos)
                v_neg = np.zeros(D, dtype=np.int64)
                v_neg[i] = -2
                fallback.append(v_neg)
            return np.array(fallback, dtype=np.int64)
            
    return result

# EVOLVE-BLOCK-END
