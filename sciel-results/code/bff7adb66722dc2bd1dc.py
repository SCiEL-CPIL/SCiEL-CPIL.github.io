import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 48 decomposition for 4x4 matrix multiplication
    # Based on the tensor product of the rank-7 decomposition of 2x2 matrix multiplication
    # with one term removed.
    
    # The 7 terms for 2x2 matrix multiplication
    # u, v, w are vectors of length 4
    # Convention: W[k*n+i] for 2x2 is W[k*2+i]
    
    u2 = [
        [1, 0, 0, 1], # P1
        [0, 1, 0, 1], # P2
        [1, 0, 1, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    
    v2 = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [0, 1, -1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    w2 = [
        [1, 0, 0, 1], # C11, C22
        [0, 0, 1, 0], # C21
        [0, 1, 0, 0], # C12
        [0, 0, 0, 1], # C22
        [0, 0, 1, 0], # C21
        [0, 0, 1, 1], # C21, C22
        [1, 0, 0, 1]  # C11, C22
    ]
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(7):
        for j in range(7):
            if i == 6 and j == 6:
                continue
            
            # Tensor product of u2[i] and u2[j]
            # u2[i] is length 4, u2[j] is length 4
            # Result is length 16
            # Index mapping: A_{k1 i1, k2 i2} -> index (k1*2+k2)*4 + (i1*2+i2) ?
            # No, standard row-major flattening of 4x4 matrix:
            # A_{r, c} where r = r1*2 + r2, c = c1*2 + c2
            # Index = r*4 + c = (r1*2+r2)*4 + (c1*2+c2)
            
            # u2[i] corresponds to A^{(1)}_{r1, c1}
            # u2[j] corresponds to A^{(2)}_{r2, c2}
            # The tensor product u2[i] \otimes u2[j] has non-zero at (r1, c1, r2, c2)
            # which maps to r = r1*2 + r2, c = c1*2 + c2
            # Index = (r1*2 + r2)*4 + (c1*2 + c2)
            
            for r1 in range(2):
                for c1 in range(2):
                    val1 = u2[i][r1 * 2 + c1]
                    if val1 == 0: continue
                    for r2 in range(2):
                        for c2 in range(2):
                            val2 = u2[j][r2 * 2 + c2]
                            if val2 == 0: continue
                            r = r1 * 2 + r2
                            c = c1 * 2 + c2
                            idx = r * 4 + c
                            U[idx][col] += val1 * val2
            
            for r1 in range(2):
                for c1 in range(2):
                    val1 = v2[i][r1 * 2 + c1]
                    if val1 == 0: continue
                    for r2 in range(2):
                        for c2 in range(2):
                            val2 = v2[j][r2 * 2 + c2]
                            if val2 == 0: continue
                            r = r1 * 2 + r2
                            c = c1 * 2 + c2
                            idx = r * 4 + c
                            V[idx][col] += val1 * val2
            
            for r1 in range(2):
                for c1 in range(2):
                    val1 = w2[i][r1 * 2 + c1]
                    if val1 == 0: continue
                    for r2 in range(2):
                        for c2 in range(2):
                            val2 = w2[j][r2 * 2 + c2]
                            if val2 == 0: continue
                            r = r1 * 2 + r2
                            c = c1 * 2 + c2
                            idx = r * 4 + c
                            W[idx][col] += val1 * val2
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
