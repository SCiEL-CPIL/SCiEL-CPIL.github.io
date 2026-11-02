"""
Tensor decomposition for 2x4 by 4x4 matrix multiplication.
Uses a structured rank-28 decomposition by exploiting block structure
and Strassen's algorithm for 2x2x2 sub-problems.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-28 exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x4 matrix.
    
    Strategy:
    1. Split A (2x4) into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3).
    2. Split B (4x4) into four 2x2 blocks:
       B_LL (rows 0-1, cols 0-1), B_LR (rows 0-1, cols 2-3)
       B_RL (rows 2-3, cols 0-1), B_RR (rows 2-3, cols 2-3)
       
       C = A @ B.
       C[:, 0:2] = A_L @ B_LL + A_R @ B_RL
       C[:, 2:4] = A_L @ B_LR + A_R @ B_RR
       
       So we have 4 products of 2x2 matrices.
    3. Each 2x2 @ 2x2 product is computed with rank 7 (Strassen).
    4. Total rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen coefficients for 2x2x2
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    blocks = [
        {'A_offset': 0, 'B_row_offset': 0, 'B_col_offset': 0, 'C_col_offset': 0},
        {'A_offset': 2, 'B_row_offset': 2, 'B_col_offset': 0, 'C_col_offset': 0},
        {'A_offset': 0, 'B_row_offset': 0, 'B_col_offset': 2, 'C_col_offset': 2},
        {'A_offset': 2, 'B_row_offset': 2, 'B_col_offset': 2, 'C_col_offset': 2},
    ]
    
    current_rank = 0
    for block in blocks:
        A_off = block['A_offset']
        B_row_off = block['B_row_offset']
        B_col_off = block['B_col_offset']
        C_col_off = block['C_col_offset']
        
        for term in strassen_terms:
            coeff_A, coeff_B, coeff_C = term
            
            for loc_idx, val in coeff_A.items():
                li, lj = divmod(loc_idx, 2)
                u_idx = li * 4 + (A_off + lj)
                U[u_idx][current_rank] += val
            
            for loc_idx, val in coeff_B.items():
                li, lj = divmod(loc_idx, 2)
                v_idx = (B_row_off + li) * 4 + (B_col_off + lj)
                V[v_idx][current_rank] += val
            
            for loc_idx, val in coeff_C.items():
                li, lj = divmod(loc_idx, 2)
                w_idx = (C_col_off + lj) * 2 + li
                W[w_idx][current_rank] += val
            
            current_rank += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
