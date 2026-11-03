import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Max disjoint 2x2x2 Strassen blocks
    n_blocks = (n // 2) * (m // 2) * (p // 2)
    rank = rank_schoolbook - n_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # For block at (ri, ci, oi):
    # A submatrix: rows ri,ri+1; cols ci,ci+1
    # B submatrix: rows ci,ci+1; cols oi,oi+1
    # C submatrix: rows oi,oi+1; cols ri,ri+1
    
    # 7 Strassen multiplications:
    # M1 = (A00+A11)(B01+B11) -> contributes to C00(+), C01(+), C11(+)
    # M2 = (A10+A11)B00       -> contributes to C10(+), C11(-)
    # M3 = A00(B10-B11)       -> contributes to C01(+), C11(+)
    # M4 = A11(B00-B10)       -> contributes to C00(+), C10(+)
    # M5 = (A00+A10)B11       -> contributes to C00(-), C11(+)
    # M6 = (A10-A00)(B00+B10) -> contributes to C00(+)
    # M7 = (A01-A11)(B01+B11) -> contributes to C00(+)
    
    # Verify: C00 = M1+M4-M5+M7, C01 = M1+M3, C10 = M2+M4, C11 = M1-M2+M3+M5
    
    strassen_defs = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A: (dr, dc, coeff), B: (dr, dc, coeff), C: (dr, dc, coeff)
        # A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
        # B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)
        # C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
        #
        # M1 = (A00+A11)(B01+B11), C: C00+, C01+, C11+
        ([((0,0,1), (1,1,1)), ((0,1,1), (1,1,1)), ((0,0,1), (0,1,1), (1,1,1))]),
        # M2 = (A10+A11)B00, C: C10+, C11-
        ([((1,0,1), (1,1,1)), ((0,0,1)), ((1,0,1), (1,1,-1))]),
        # M3 = A00(B10-B11), C: C01+, C11+
        ([((0,0,1)), ((1,0,1), (1,1,-1)), ((0,1,1), (1,1,1))]),
        # M4 = A11(B00-B10), C: C00+, C10+
        ([((1,1,1)), ((0,0,1), (1,0,-1)), ((0,0,1), (1,0,1))]),
        # M5 = (A00+A10)B11, C: C00-, C11+
        ([((0,0,1), (1,0,1)), ((1,1,1)), ((0,0,-1), (1,1,1))]),
        # M6 = (A10-A00)(B00+B10), C: C00+
        ([((1,0,1), (0,0,-1)), ((0,0,1), (1,0,1)), ((0,0,1))]),
        # M7 = (A01-A11)(B01+B11), C: C00+
        ([((0,1,1), (1,1,-1)), ((0,1,1), (1,1,1)), ((0,0,1))]),
    ]
    
    # Enumerate all disjoint 2x2x2 blocks
    for ri in range(0, n - 1, 2):
        for ci in range(0, m - 1, 2):
            for oi in range(0, p - 1, 2):
                for A_coeffs, B_coeffs, C_coeffs in strassen_defs:
                    for (dr, dc, coeff) in A_coeffs:
                        U[(ri + dr) * m + (ci + dc)][col] = coeff
                    for (dr, dc, coeff) in B_coeffs:
                        V[(ci + dr) * p + (oi + dc)][col] = coeff
                    for (dr, dc, coeff) in C_coeffs:
                        W[(oi + dr) * n + (ri + dc)][col] = coeff
                    col += 1
    
    # Schoolbook for remaining elements
    covered = set()
    for ri in range(0, n - 1, 2):
        for ci in range(0, m - 1, 2):
            for oi in range(0, p - 1, 2):
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered.add((ri + di, ci + dj, oi + dk))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    return {"U": U, "V": V, "W": W}
