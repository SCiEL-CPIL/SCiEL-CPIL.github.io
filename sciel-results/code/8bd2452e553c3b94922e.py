"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS  # 2, 4, 8

    # We'll build a rank-56 decomposition using Strassen on 2x2 blocks.
    # A is 2x4, B is 4x8.
    # Split A into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3).
    # Split B into 4 blocks of 4x2: B_0 (cols 0-1), B_1 (cols 2-3), B_2 (cols 4-5), B_3 (cols 6-7).
    # Each B_k splits into B_k_top (2x2, rows 0-1) and B_k_bot (2x2, rows 2-3).
    # C[:, k:k+2] = A_L @ B_k_top + A_R @ B_k_bot
    # Each 2x2 product uses Strassen (rank 7), so per B_k: 2*7 = 14, total 4*14 = 56.

    rank = 56
    # U: shape (n*m, rank) = (8, 56)
    # V: shape (m*p, rank) = (32, 56)
    # W: shape (p*n, rank) = (16, 56)
    # Indexing: U[i,j], V[j,k], W[k,i] where i in [0,8), j in [0,32), k in [0,16)
    # Actually: U[i, r], V[j, r], W[k, r] for rank r
    # i = row_in_A * m + col_in_A = row * 4 + col, so i in [0, 8)
    # j = row_in_B * p + col_in_B = row * 8 + col, so j in [0, 32)
    # k = col_in_C * n + row_in_C = col * 2 + row, so k in [0, 16)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 2x2x2: 7 products
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    strassen_defs = [
        # (U_dict, V_dict, W_dict)
        # M1 = (A11+A22)(B11+B22)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):-1}),
        # M2 = (A21+A22)B11
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,0):-1, (1,0):1}),
        # M3 = A11(B21-B22)
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A22(B11-B21)
        ({(1,1):1}, {(0,0):1, (1,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A11+A12)B22
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A21-A11)(B11+B12)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A12-A22)(B21+B22)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,1):1}),
    ]

    col = 0
    for b_col in range(0, p, 2):  # b_col = 0, 2, 4, 6
        # Block for A_L (cols 0-1 of A) * B_k_top (rows 0-1 of B)
        a_row_off = 0
        a_col_off = 0
        b_row_off = 0
        for uc, vc, wc in strassen_defs:
            for (ri, cj), coef in uc.items():
                U[(a_row_off + ri) * m + (a_col_off + cj), col] = coef
            for (rj, ck), coef in vc.items():
                V[(b_row_off + rj) * p + (b_col + ck), col] = coef
            for (rk, ri2), coef in wc.items():
                W[(b_col + rk) * n + (a_row_off + ri2), col] = coef
            col += 1
        
        # Block for A_R (cols 2-3 of A) * B_k_bot (rows 2-3 of B)
        a_row_off = 0
        a_col_off = 2
        b_row_off = 2
        for uc, vc, wc in strassen_defs:
            for (ri, cj), coef in uc.items():
                U[(a_row_off + ri) * m + (a_col_off + cj), col] = coef
            for (rj, ck), coef in vc.items():
                V[(b_row_off + rj) * p + (b_col + ck), col] = coef
            for (rk, ri2), coef in wc.items():
                W[(b_col + rk) * n + (a_row_off + ri2), col] = coef
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
