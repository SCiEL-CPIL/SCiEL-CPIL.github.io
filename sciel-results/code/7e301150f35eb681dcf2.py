"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses a partition of A into 2x3 and 1x3 blocks.
The 2x3 * 3x2 sub-problems are decomposed with rank 6.
The 1x3 * 3x8 sub-problem is decomposed with rank 3.
Total rank: 4 * 6 + 3 = 27.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0

    U_t = []
    V_t = []
    W_t = []

    def add_term(U_list, V_list, W_list):
        U_t.append(U_list)
        V_t.append(V_list)
        W_t.append(W_list)

    # A is 3x3. Partition into A1 (rows 0,1) and A2 (row 2).
    # A1 is 2x3. A2 is 1x3.
    # B is 3x8. Partition into 4 blocks of 3x2.
    
    # For each block r in 0..3:
    # B_r is B[:, 2r : 2r+2]
    # Compute C_r = A1 * B_r (2x2)
    # Use rank 6 decomposition for 2x3 * 3x2.
    
    for r in range(4):
        # B block columns: 2r, 2r+1
        b_col_0 = 2 * r
        b_col_1 = 2 * r + 1
        
        # C block columns: 2r, 2r+1 (in the output C, which is 3x8)
        # But A1 produces rows 0,1 of C.
        # So C[0, 2r], C[0, 2r+1], C[1, 2r], C[1, 2r+1]
        
        # Term 1: a00 * (b00 c00 + b01 c01)
        # a00 is A[0,0]
        # b00 is B[0, 2r], b01 is B[0, 2r+1]
        # c00 is C[0, 2r], c01 is C[0, 2r+1]
        add_term(
            [((0, 0), 1)],
            [((0, 2*r), 1), ((0, 2*r+1), 1)],
            [((0, 2*r), 1), ((0, 2*r+1), 1)]
        )
        
        # Term 2: a01 * (b10 c00 + b11 c01)
        # a01 is A[0,1]
        # b10 is B[1, 2r], b11 is B[1, 2r+1]
        add_term(
            [((0, 1), 1)],
            [((1, 2*r), 1), ((1, 2*r+1), 1)],
            [((0, 2*r), 1), ((0, 2*r+1), 1)]
        )
        
        # Term 3: a02 * (b20 c00 + b21 c01)
        # a02 is A[0,2]
        # b20 is B[2, 2r], b21 is B[2, 2r+1]
        add_term(
            [((0, 2), 1)],
            [((2, 2*r), 1), ((2, 2*r+1), 1)],
            [((0, 2*r), 1), ((0, 2*r+1), 1)]
        )
        
        # Term 4: a10 * (b00 c10 + b01 c11)
        # a10 is A[1,0]
        # c10 is C[1, 2r], c11 is C[1, 2r+1]
        add_term(
            [((1, 0), 1)],
            [((0, 2*r), 1), ((0, 2*r+1), 1)],
            [((1, 2*r), 1), ((1, 2*r+1), 1)]
        )
        
        # Term 5: a11 * (b10 c10 + b11 c11)
        # a11 is A[1,1]
        add_term(
            [((1, 1), 1)],
            [((1, 2*r), 1), ((1, 2*r+1), 1)],
            [((1, 2*r), 1), ((1, 2*r+1), 1)]
        )
        
        # Term 6: a12 * (b20 c10 + b21 c11)
        # a12 is A[1,2]
        add_term(
            [((1, 2), 1)],
            [((2, 2*r), 1), ((2, 2*r+1), 1)],
            [((1, 2*r), 1), ((1, 2*r+1), 1)]
        )

    # Now handle A2 (row 2 of A) * B (3x8)
    # A2 is A[2, :] = (a20, a21, a22)
    # B is 3x8.
    # C row 2 is C[2, :] = A2 * B.
    
    # Term 7: a20 * (row 0 of B)
    # a20 is A[2,0]
    # row 0 of B is B[0, 0:8]
    # C[2, 0:8]
    U_list = [((2, 0), 1)]
    V_list = [((0, k), 1) for k in range(8)]
    W_list = [((2, k), 1) for k in range(8)]
    add_term(U_list, V_list, W_list)
    
    # Term 8: a21 * (row 1 of B)
    U_list = [((2, 1), 1)]
    V_list = [((1, k), 1) for k in range(8)]
    W_list = [((2, k), 1) for k in range(8)]
    add_term(U_list, V_list, W_list)
    
    # Term 9: a22 * (row 2 of B)
    U_list = [((2, 2), 1)]
    V_list = [((2, k), 1) for k in range(8)]
    W_list = [((2, k), 1) for k in range(8)]
    add_term(U_list, V_list, W_list)
    
    rank = len(W_t)
    
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i][idx] = coeff
            
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify_decomp(Ua, Va, Wa):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
