"""Rank-63 decomposition for 6x6 matrix multiplication using Kronecker product of 2x2 Strassen and 3x3 schoolbook."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # all 6
    
    # 2x2 Strassen coefficients (rank 7)
    # Each term: (u2, v2, w2) where u2, v2, w2 are 2x2 matrices (flattened to length 4)
    # Indexing: [r0c0, r0c1, r1c0, r1c1]
    
    strassen_2x2 = [
        # (u2, v2, w2)
        # Term 1: (A00+A11)(B00+B11) -> C00, C11
        (
            [1, 0, 0, 1],   # u2
            [1, 0, 0, 1],   # v2
            [1, 0, 0, 1],   # w2: C00 += 1, C11 += 1
        ),
        # Term 2: (A10+A11)B00 -> C10, C11
        (
            [0, 0, 1, 1],   # u2
            [1, 0, 0, 0],   # v2
            [0, 0, 1, -1],  # w2: C10 += 1, C11 -= 1
        ),
        # Term 3: A00(B01-B11) -> C01, C11
        (
            [1, 0, 0, 0],   # u2
            [0, 1, 0, -1],  # v2
            [0, 1, 0, 1],   # w2: C01 += 1, C11 += 1
        ),
        # Term 4: A11(B10-B00) -> C00, C10
        (
            [0, 0, 0, 1],   # u2
            [0, 0, 1, -1],  # v2
            [1, 0, 1, 0],   # w2: C00 += 1, C10 += 1
        ),
        # Term 5: (A00+A01)B11 -> C00, C01
        (
            [1, 1, 0, 0],   # u2
            [0, 0, 0, 1],   # v2
            [-1, 1, 0, 0],  # w2: C00 -= 1, C01 += 1
        ),
        # Term 6: (A10-A00)(B00+B01) -> C11
        (
            [0, 0, 1, -1],  # u2
            [1, 1, 0, 0],   # v2
            [0, 0, 0, 1],   # w2: C11 += 1
        ),
        # Term 7: (A01-A11)(B10+B11) -> C00
        (
            [0, 1, 0, -1],  # u2
            [0, 0, 1, 1],   # v2
            [1, 0, 0, 0],   # w2: C00 += 1
        ),
    ]
    
    # 3x3 schoolbook coefficients (rank 9)
    # Each term: (u3, v3, w3) where u3, v3, w3 are 3x3 matrices (flattened to length 9)
    # Indexing: [r0c0, r0c1, r0c2, r1c0, r1c1, r1c2, r2c0, r2c1, r2c2]
    
    schoolbook_3x3 = []
    for i in range(3):
        for k in range(3):
            u3 = [0] * 9
            u3[i * 3 + k] = 1
            v3 = [0] * 9
            v3[k * 3 + j] = 1  # placeholder, will fix below
            w3 = [0] * 9
            w3[i * 3 + j] = 1  # placeholder
            # Actually, for schoolbook: C[i,j] += A[i,k] * B[k,j]
            # So u3 has 1 at (i,k), v3 has 1 at (k,j), w3 has 1 at (i,j)
            for j in range(3):
                u3_t = [0] * 9
                u3_t[i * 3 + k] = 1
                v3_t = [0] * 9
                v3_t[k * 3 + j] = 1
                w3_t = [0] * 9
                w3_t[i * 3 + j] = 1
                schoolbook_3x3.append((u3_t, v3_t, w3_t))
    
    # Now combine: rank = 7 * 9 = 63
    rank = 63
    U = [[0] * rank for _ in range(n * m)]  # 36 x 63
    V = [[0] * rank for _ in range(m * p)]  # 36 x 63
    W = [[0] * rank for _ in range(p * n)]  # 36 x 63
    
    col = 0
    for s2 in range(7):
        u2, v2, w2 = strassen_2x2[s2]
        for s3 in range(9):
            u3, v3, w3 = schoolbook_3x3[s3]
            
            # U: 6x6 matrix, index I*6 + K
            # I = 2*r2 + i3, K = 2*c2 + k3
            for r2 in range(2):
                for c2 in range(2):
                    val_u2 = u2[r2 * 2 + c2]
                    if val_u2 == 0:
                        continue
                    for i3 in range(3):
                        for k3 in range(3):
                            val_u3 = u3[i3 * 3 + k3]
                            if val_u3 == 0:
                                continue
                            I = 2 * r2 + i3
                            K = 2 * c2 + k3
                            U[I * 6 + K][col] += val_u2 * val_u3
            
            # V: 6x6 matrix, index K*6 + J
            # K = 2*c2 + k3, J = 2*c2p + j3
            for c2 in range(2):
                for c2p in range(2):
                    val_v2 = v2[c2 * 2 + c2p]
                    if val_v2 == 0:
                        continue
                    for k3 in range(3):
                        for j3 in range(3):
                            val_v3 = v3[k3 * 3 + j3]
                            if val_v3 == 0:
                                continue
                            K = 2 * c2 + k3
                            J = 2 * c2p + j3
                            V[K * 6 + J][col] += val_v2 * val_v3
            
            # W: 6x6 matrix, index J*6 + I
            # J = 2*c2p + j3, I = 2*r2 + i3
            for c2p in range(2):
                for r2 in range(2):
                    val_w2 = w2[c2p * 2 + r2]
                    if val_w2 == 0:
                        continue
                    for j3 in range(3):
                        for i3 in range(3):
                            val_w3 = w3[j3 * 3 + i3]
                            if val_w3 == 0:
                                continue
                            J = 2 * c2p + j3
                            I = 2 * r2 + i3
                            W[J * 6 + I][col] += val_w2 * val_w3
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
