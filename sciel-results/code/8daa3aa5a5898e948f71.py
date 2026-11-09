import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    # Generate 240 minimal vectors of E8 scaled by 2 (integer coordinates)
    # E8 minimal vectors have norm squared 2. Scaled by 2, norm squared is 8.
    # There are 112 integer type and 128 half-integer type.
    # Scaled by 2:
    # Integer type: 2 * (+/- e_i +/- e_j) -> (+/- 2, +/- 2, 0...)
    # Half-integer type: 2 * (+/- 0.5...) -> (+/- 1, +/- 1...) with even sum?
    # Actually, just generate all 240 and scale.
    
    vecs = []
    
    # Type 1: 2 * (+/- e_i +/- e_j)
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(8, dtype=np.int64)
                    v[i] = 2 * s1
                    v[j] = 2 * s2
                    vecs.append(v)
                    
    # Type 2: 2 * (half-integer vectors)
    # Half-integer vectors are (+/- 0.5, ..., +/- 0.5) with even sum of coords.
    # Scaled by 2: (+/- 1, ..., +/- 1) with even sum of coords.
    # Sum of coords must be even.
    # All coords are +/- 1. Sum is even iff number of +1s is even.
    # Number of +1s can be 0, 2, 4, 6, 8.
    from itertools import combinations, product
    
    for k_plus in [0, 2, 4, 6, 8]:
        for indices in combinations(range(8), k_plus):
            # Create vector with 1 at indices, -1 elsewhere
            v = np.ones(8, dtype=np.int64)
            v[list(indices)] = 1
            # Wait, if k_plus is number of +1s, then rest are -1s.
            # Sum = k_plus - (8-k_plus) = 2*k_plus - 8. Even.
            # This generates all such vectors.
            vecs.append(v)
            
    # Total should be 112 + 128 = 240
    S8 = np.array(vecs, dtype=np.int64)
    
    # Generate 4 vectors in Z^3 with norm squared 4 and mutual dist >= 16?
    # No, for M_3=4, we needed D_3 >= 12.
    # Vectors: +/- 2e_i.
    # We pick a pair: 2e_1, -2e_1. Dist 16.
    # Or we can pick 4 vectors if we use M_3=2?
    # For M_3=2, we needed D_3 >= 10.
    # Norm 2 vectors in Z^3: (+/- 1, +/- 1, 0).
    # Max dist 8. 8 < 10. So M_3=2 fails for scaled E8.
    
    # So we must use M_3=4.
    # Vectors: +/- 2e_1, +/- 2e_2, +/- 2e_3.
    # Pick 2e_1 and -2e_1.
    S3 = np.array([[2, 0, 0], [-2, 0, 0]], dtype=np.int64)
    
    # Cartesian product
    # S8 shape (240, 8), S3 shape (2, 3)
    # Result shape (480, 11)
    
    # Pad S8 with 0s to dim 11
    S8_padded = np.hstack([S8, np.zeros((S8.shape[0], 3), dtype=np.int64)])
    
    # Tile S3 to match S8 count
    S3_tiled = np.tile(S3, (S8.shape[0], 1))
    
    points = np.hstack([S8_padded, S3_tiled])
    
    return points
