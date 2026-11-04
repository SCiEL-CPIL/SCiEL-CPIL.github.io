import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)

    # Rank 56 decomposition
    # 8 blocks of 2x2 * 2x2 Strassen (rank 7 each)
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen coefficients for 2x2 * 2x2
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # M1=(a0+a3)(b0+b3)
    # M2=(a2+a3)b0
    # M3=a0(b1-b3)
    # M4=a3(b2-b0)
    # M5=(a0+a1)b3
    # M6=(a2-a0)(b0+b1)
    # M7=(a1-a3)(b2+b3)
    # C00=M1+M4-M5+M7
    # C01=M3+M5
    # C10=M2+M4
    # C11=M1-M2+M3+M6

    # Local indices for 2x2 block:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    u_coeffs = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    v_coeffs = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    w_coeffs = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1],
        [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    # 8 blocks:
    # t=0..3: A_L (cols 0,1), B_T (rows 0,1), B cols 0,1 / 2,3 / 4,5 / 6,7
    # t=4..7: A_R (cols 2,3), B_B (rows 2,3), B cols 0,1 / 2,3 / 4,5 / 6,7

    for t in range(8):
        if t < 4:
            a_col_off = 0
            b_row_off = 0
        else:
            a_col_off = 2
            b_row_off = 2
            
        b_col_off = (t % 4) * 2

        # U indices: A[i,j] is at i*m + j
        # Local a0..a3 correspond to:
        # a0: A[0, a_col_off]
        # a1: A[0, a_col_off+1]
        # a2: A[1, a_col_off]
        # a3: A[1, a_col_off+1]
        u_idx = [
            0 * m + (0 + a_col_off),
            0 * m + (1 + a_col_off),
            1 * m + (0 + a_col_off),
            1 * m + (1 + a_col_off)
        ]
        
        # V indices: B[i,j] is at i*p + j
        # Local b0..b3 correspond to:
        # b0: B[b_row_off, b_col_off]
        # b1: B[b_row_off, b_col_off+1]
        # b2: B[b_row_off+1, b_col_off]
        # b3: B[b_row_off+1, b_col_off+1]
        v_idx = [
            (0 + b_row_off) * p + (0 + b_col_off),
            (0 + b_row_off) * p + (1 + b_col_off),
            (1 + b_row_off) * p + (0 + b_col_off),
            (1 + b_row_off) * p + (1 + b_col_off)
        ]
        
        # W indices: C[i,k] is at k*n + i
        # Local c0..c3 correspond to:
        # c0: C[0, b_col_off]
        # c1: C[0, b_col_off+1]
        # c2: C[1, b_col_off]
        # c3: C[1, b_col_off+1]
        
        # w_idx order must match w_coeffs order:
        # w_coeffs are for C00, C01, C10, C11?
        # Let's check w_coeffs[0] = [1, 0, 0, 1].
        # This adds to local index 0 and 3.
        # In Strassen, M1 adds to C00 and C11.
        # So local index 0 must be C00 and local index 3 must be C11.
        # My mapping:
        # 0: C[0, b_col_off] -> C00
        # 1: C[0, b_col_off+1] -> C01
        # 2: C[1, b_col_off] -> C10
        # 3: C[1, b_col_off+1] -> C11
        # This is consistent.
        
        w_idx = [
            (0 + b_col_off) * n + 0,
            (1 + b_col_off) * n + 0,
            (0 + b_col_off) * n + 1,
            (1 + b_col_off) * n + 1
        ]
        
        for l in range(7):
            r = t * 7 + l
            for li, c in enumerate(u_coeffs[l]):
                if c:
                    U[u_idx[li], r] += c
            for li, c in enumerate(v_coeffs[l]):
                if c:
                    V[v_idx[li], r] += c
            for li, c in enumerate(w_coeffs[l]):
                if c:
                    W[w_idx[li], r] += c

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
