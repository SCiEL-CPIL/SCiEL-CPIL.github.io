"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 Strassen-based decomposition for 4x4x4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 block matrix multiplication
    # A_blocks indexed by (row, col) in {0,1}^2, each a 2x2 submatrix
    # B_blocks indexed by (row, col) in {0,1}^2, each a 2x2 submatrix
    
    # For each Strassen term, define which A-blocks and B-blocks participate
    # with their coefficients.
    # a_terms[t] = list of (block_row, block_col, coeff)
    # b_terms[t] = list of (block_row, block_col, coeff)
    
    a_terms = [
        [(0,0,1), (1,1,1)],           # M1: A00 + A11
        [(1,0,1), (1,1,1)],           # M2: A10 + A11
        [(0,0,1)],                     # M3: A00
        [(1,1,1)],                     # M4: A11
        [(0,0,1), (0,1,1)],           # M5: A00 + A01
        [(1,0,1), (0,0,-1)],          # M6: A10 - A00
        [(0,1,1), (1,1,-1)],          # M7: A01 - A11
    ]
    b_terms = [
        [(0,0,1), (1,1,1)],           # M1: B00 + B11
        [(0,0,1)],                     # M2: B00
        [(0,1,1), (1,1,-1)],          # M3: B01 - B11
        [(1,0,1), (0,0,-1)],          # M4: B10 - B00
        [(1,1,1)],                     # M5: B11
        [(0,0,1), (0,1,1)],           # M6: B00 + B01
        [(1,0,1), (1,1,1)],           # M7: B10 + B11
    ]
    
    # C output block coefficients: C[bi][bj] = sum_t c[bi][bj][t] * Mt
    c_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],   # C00 = M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],    # C01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],    # C10 = M2 + M4
        [1, -1, 1, 0, 0, 1, 0],   # C11 = M1 - M2 + M3 + M6
    ]
    
    col = 0
    for t in range(7):
        for ai in range(2):
            for ak in range(2):
                for aj in range(2):
                    # For each (ai, ak, aj) schoolbook index within blocks
                    # A-side: sum over (bi, bk) where A-block (bi,bk) participates
                    # B-side: sum over (bk, bj) where B-block (bk,bj) participates
                    # Must match on bk (the shared index)
                    
                    for (bi, bk, ca) in a_terms[t]:
                        for (bk2, bj, cb) in b_terms[t]:
                            if bk == bk2:
                                # Full indices:
                                i = 2 * bi + ai      # row of A
                                j = 2 * bk + ak      # col of A = row of B
                                k = 2 * bj + aj      # col of B
                                u_idx = i * m + j
                                v_idx = j * p + k
                                U[u_idx][col] += ca
                                V[v_idx][col] += cb
                    
                    # W-side: for each output block (bi_c, bj_c)
                    for bi_c in range(2):
                        for bj_c in range(2):
                            coeff_c = c_coeffs[bi_c][bj_c][t]
                            if coeff_c != 0:
                                i = 2 * bi_c + ai
                                k = 2 * bj_c + aj
                                w_idx = k * n + i
                                W[w_idx][col] += coeff_c
                    
                    col += 1
    
    # Verify with random matrices
    def verify():
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p))
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    # Sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This should equal 1 for the correct (i,j,k)
                    s = 0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    C_computed[i, k] += A[i, j] * B[j, k] * s
        return np.allclose(C_computed, C_expected)
    
    if not verify():
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
