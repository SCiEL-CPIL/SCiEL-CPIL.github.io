import numpy as np
from itertools import combinations

D = 11

def construct():
    # Generate all vectors with exactly four non-zero entries, each +/- 1
    # This corresponds to the set of 4-sparse vectors.
    # Norm squared is 4.
    # Min pairwise distance squared is at least 4 (if sharing 3 indices, diff in 1 -> 4; if sharing 2, diff in 2 -> 4; etc).
    # Wait, if sharing 3 indices:
    # v = (1, 1, 1, 1, 0...)
    # w = (1, 1, 1, -1, 0...)
    # Dist sq = 0 + 0 + 0 + 4 = 4.
    # 4 >= 4. OK.
    # If sharing 2 indices:
    # v = (1, 1, 1, 1, 0...)
    # w = (1, 1, 0, 0, 1, 1, 0...)
    # Dist sq = 0 + 0 + 1 + 1 + 1 + 1 = 4.
    # 4 >= 4. OK.
    # So all 4-sparse vectors are valid.
    
    vectors = []
    # Indices for the four non-zero positions
    for combo in combinations(range(D), 4):
        # Generate all 2^4 sign combinations
        for mask in range(16):
            v = np.zeros(D, dtype=np.int64)
            for k, idx in enumerate(combo):
                if (mask >> k) & 1:
                    v[idx] = -1
                else:
                    v[idx] = 1
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
