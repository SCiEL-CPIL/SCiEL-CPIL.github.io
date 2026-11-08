"""Strassen-based rank-189 decomposition for 6x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # all 6
    
    # 2x2 Strassen decomposition (rank 7)
    # For multiplying two 2x2 matrices A, B -> C
    # The 7 products are:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen 2x2: A is 2x2, B is 2x2
    # A indices: (i, k) -> A[i,k] for i,k in {0,1}
    # B indices: (k, j) -> B[k,j] for k,j in {0,1}
    # C indices: (i, j) -> C[i,j] for i,j in {0,1}
    
    # For each of the 7 Strassen multiplications, define:
    # - U coefficients: linear combination of A entries
    # - V coefficients: linear combination of B entries
    # - W coefficients: how the product contributes to C entries
    
    # Strassen forms (as dictionaries mapping (i,k) -> coeff for A, (k,j) -> coeff for B)
    strassen_A = [
        {(0,0): 1, (1,1): 1},          # M1: (A00+A11)
        {(1,0): 1, (1,1): 1},          # M2: (A10+A11)
        {(0,0): 1},                     # M3: A00
        {(1,1): 1},                     # M4: A11
        {(0,0): 1, (0,1): 1},          # M5: (A00+A01)
        {(1,0): 1, (0,0): -1},         # M6: (A10-A00)
        {(0,1): 1, (1,1): -1},         # M7: (A01-A11)
    ]
    
    strassen_B = [
        {(0,0): 1, (1,1): 1},          # M1: *(B00+B11)
        {(0,0): 1},                     # M2: *B00
        {(0,1): 1, (1,1): -1},         # M3: *(B01-B11)
        {(1,0): 1, (0,0): -1},         # M4: *(B10-B00)
        {(1,1): 1},                     # M5: *B11
        {(0,0): 1, (0,1): 1},          # M6: *(B00+B01)
        {(1,0): 1, (1,1): 1},          # M7: *(B10+B11)
    ]
    
    # Reconstruction: for each Strassen product Mt, which C entries does it contribute to?
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    strassen_W = [
        # M1: contributes to C00 (+1) and C11 (+1)
        [(0,0), 1], [(1,1), 1],
    ]
    strassen_W = [
        # M1
        [((0,0), 1), ((1,1), 1)],
        # M2: contributes to C10 (+1) and C11 (-1)
        [((1,0), 1), ((1,1), -1)],
        # M3: contributes to C01 (+1) and C11 (+1)
        [((0,1), 1), ((1,1), 1)],
        # M4: contributes to C00 (+1) and C10 (+1)
        [((0,0), 1), ((1,0), 1)],
        # M5: contributes to C00 (-1) and C01 (+1)
        [((0,0), -1), ((0,1), 1)],
        # M6: contributes to C11 (+1)
        [((1,1), 1)],
        # M7: contributes to C00 (+1)
        [((0,0), 1)],
    ]
    
    # Now tile 6x6 with 2x2 blocks: 3x3 blocks
    # For each block position (bi, bj) where bi,bj in {0,1,2}:
    #   The submatrix A[2bi:2bi+2, 2bj:2bj+2] * B[2bj:2bj+2, 2bk:2bk+2]
    #   contributes to C[2bi:2bi+2, 2bk:2bk+2]
    # 
    # For each block (bi, bj) pair and each Strassen multiplication t:
    #   We get one rank-1 factor
    
    rank = 9 * 7  # 9 block positions * 7 Strassen mults = 63... wait
    # Actually: for each (bi, bk) block of the result, we need to multiply
    # A[2bi:2bi+2, 2bj:2bj+2] * B[2bj:2bj+2, 2bk:2bk+2] for all bj
    # and sum over bj. But Strassen decomposes each 2x2 multiply into 7 mults.
    # So for each (bi, bk) pair (3x3 = 9 pairs) and each bj (3 values) and each t (7):
    #   That's 9 * 3 * 7 = 189 rank-1 factors
    
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]  # (36, 189)
    V = [[0] * rank for _ in range(m * p)]  # (36, 189)
    W = [[0] * rank for _ in range(p * n)]  # (36, 189)
    
    col = 0
    for bi in range(3):       # row block of A / row block of C
        for bj in range(3):   # col block of A / row block of B
            for bk in range(3):  # col block of B / col block of C
                for t in range(7):
                    # U: A entries
                    # A is indexed as [i*m + k] where i is row, k is col in the 6x6
                    # Wait, the convention: U has shape (n*m, rank), V has shape (m*p, rank)
                    # So U[i*m + k, c] corresponds to A[i, k]
                    # V[k*p + j, c] corresponds to B[k, j]
                    # W[j*n + i, c] corresponds to C[i, j] (trace convention: W[k*n+i, r] in task)
                    
                    # For block (bi, bj) of A: rows 2bi, 2bi+1; cols 2bj, 2bj+1
                    # For block (bj, bk) of B: rows 2bj, 2bj+1; cols 2bk, 2bk+1
                    # For block (bi, bk) of C: rows 2bi, 2bi+1; cols 2bk, 2bk+1
                    
                    # A coefficients
                    for (ai, ak), coeff in strassen_A[t].items():
                        global_i = 2 * bi + ai
                        global_k = 2 * bj + ak
                        U[global_i * m + global_k][col] = coeff
                    
                    # B coefficients
                    for (bk_local, bj_local), coeff in strassen_B[t].items():
                        # B[k, j] where k is row, j is col
                        global_bk = 2 * bj + bk_local
                        global_bj = 2 * bk + bj_local
                        V[global_bk * p + global_bj][col] = coeff
                    
                    # C coefficients
                    for (ci, cj), coeff in strassen_W[t]:
                        global_i = 2 * bi + ci
                        global_j = 2 * bk + cj
                        # W[j*n + i, col]
                        W[global_j * n + global_i][col] = coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
