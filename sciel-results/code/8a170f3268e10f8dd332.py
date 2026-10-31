"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    We use the D_11 lattice construction: all vectors of the form +/-e_i +/- e_j
    for i < j. These all have norm sqrt(2), and the minimum pairwise distance
    between distinct vectors is sqrt(2) (achieved between e_i+e_j and e_i-e_j,
    or between e_i+e_j and -e_i-e_j, etc. -- actually let's verify).
    
    Wait: distance between (e_i + e_j) and (e_i - e_j) is |2e_j| = 2*sqrt(1) = 2? No.
    Actually these are different coordinate positions. Let me recompute.
    
    v1 = e_i + e_j, v2 = e_i - e_j. 
    v1 - v2 = 2e_j. Norm = 2. That's > sqrt(2). Good.
    
    v1 = e_i + e_j, v2 = -e_i + e_j.
    v1 - v2 = 2e_i. Norm = 2 > sqrt(2). Good.
    
    v1 = e_i + e_j, v2 = e_k + e_l (disjoint indices).
    v1 - v2 has four nonzero entries of magnitude 1. Norm = sqrt(4) = 2 > sqrt(2).
    
    v1 = e_i + e_j, v2 = -(e_i + e_j).
    v1 - v2 = 2e_i + 2e_j. Norm = sqrt(8) = 2*sqrt(2) > sqrt(2).
    
    So all pairwise distances are >= sqrt(2), and all norms are sqrt(2).
    This satisfies the condition: min pairwise dist >= max norm.
    
    Number of such vectors: C(11,2) * 4 = 55 * 4 = 220.
    """
    deadline = time.time() + 140  # safety margin before 150s kill
    
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
    
    # Check time -- if we've gone over, return what we have
    if time.time() > deadline:
        return np.array(vectors, dtype=np.int64)
    
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
