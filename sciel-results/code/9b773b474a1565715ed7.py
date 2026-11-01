import numpy as np

def construct():
    D = 11
    # Generate all vectors in D11 lattice with norm sqrt(2)
    # These are vectors with exactly two non-zero entries, which are +/- 1.
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            # 4 sign combinations for positions i and j
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    vecs = np.array(vectors, dtype=np.int64)
    
    # Optional: Verify validity
    # Norms are all sqrt(2)
    # Min distance is sqrt(2)
    # Condition: min_dist >= max_norm => sqrt(2) >= sqrt(2) is True.
    
    return vecs
