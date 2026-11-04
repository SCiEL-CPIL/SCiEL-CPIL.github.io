"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Constructs a kissing configuration in R^11 using the direct sum of E8 and D3 lattices 
    plus mixed vectors, all scaled to have integer coordinates and norm squared 8.
    
    The set consists of:
    1. 112 vectors of type (±2, ±2, 0, ..., 0) in the first 8 dimensions.
    2. 128 vectors of type (±1, ..., ±1) in the first 8 dimensions with even parity.
    3. 12 vectors of type (±2, ±2, 0) in the last 3 dimensions.
    4. 96 vectors of type (±2, 0, ..., 0, ±2, 0, ...) with one non-zero in the first 8 
       and one in the last 3 dimensions.
    
    Total: 348 vectors.
    Norm squared of each vector is 8.
    Minimum pairwise distance squared is 8.
    Condition min_dist >= max_norm holds (sqrt(8) >= sqrt(8)).
    """
    vectors = []
    
    # 1. E8 Part Type 1: (±2, ±2, 0, ..., 0) in dims 0-7
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 11
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
                    
    # 2. E8 Part Type 2: (±1, ..., ±1) in dims 0-7 with even number of -1s
    for mask in range(256):
        minus_count = 0
        for i in range(8):
            if (mask >> i) & 1:
                minus_count += 1
        if minus_count % 2 == 0:
            v = [0] * 11
            for i in range(8):
                if (mask >> i) & 1:
                    v[i] = -1
                else:
                    v[i] = 1
            vectors.append(v)
            
    # 3. D3 Part: (±2, ±2, 0) permutations in dims 8-10
    for i in range(8, 11):
        for j in range(i + 1, 11):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 11
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
    
    # 4. Mixed Part: One non-zero in 0-7, one in 8-10
    for i in range(8):
        for j in range(8, 11):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * 11
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)

    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
