"""Matrix multiplication 2,4,6 — rank-minimal exact tensor decomposition."""

import numpy as np
import time

DIMENSIONS = [2, 4, 6]


# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 6
    
    # Dimensions for U, V, W
    dim_U = n * m  # 8
    dim_V = m * p  # 24
    dim_W = p * n  # 12
    
    # Rank 42 decomposition based on Strassen's algorithm
    # Split into 6 blocks of 2x2 x 2x2, each rank 7
    rank = 42
    
    U = np.zeros((dim_U, rank), dtype=np.float64)
    V = np.zeros((dim_V, rank), dtype=np.float64)
    W = np.zeros((dim_W, rank), dtype=np.float64)
    
    # Define the 6 instances of 2x2 matrix multiplication
    # Each instance is defined by (i_offset, j_offset, k_offset)
    # i_offset: row offset in A and C
    # j_offset: col offset in A and row offset in B
    # k_offset: col offset in B and C
    instances = [
        (0, 0, 0),  # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        (0, 0, 2),  # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        (0, 0, 4),  # A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
        (0, 2, 0),  # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        (0, 2, 2),  # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        (0, 2, 4),  # A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    ]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define terms as (name, U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (local_i, local_j, coeff)
    # V_coeffs: list of (local_j, local_k, coeff)
    # W_coeffs: list of (local_k, local_i, coeff)
    
    # Note: Local indices are 0,1
    # Global indices:
    # U index = (i_off + li) * m + (j_off + lj)
    # V index = (j_off + lj) * p + (k_off + lk)
    # W index = (k_off + lk) * n + (i_off + li)
    
    strassen_terms = [
        # M1
        (
            [(0, 0, 1), (1, 1, 1)],  # A00, A11
            [(0, 0, 1), (1, 1, 1)],  # B00, B11
            [(0, 0, 1), (1, 1, 1)]   # C00, C11
        ),
        # M2
        (
            [(1, 0, 1), (1, 1, 1)],  # A10, A11
            [(0, 0, 1)],             # B00
            [(0, 1, 1)]              # C10
        ),
        # M3
        (
            [(0, 0, 1)],             # A00
            [(0, 1, 1), (1, 1, -1)],# B01, -B11
            [(1, 0, 1)]              # C01
        ),
        # M4
        (
            [(1, 1, 1)],             # A11
            [(1, 0, 1), (0, 0, -1)],# B10, -B00
            [(0, 0, 1), (1, 0, 1)]  # C00, C10
        ),
        # M5
        (
            [(0, 0, 1), (0, 1, 1)],  # A00, A01
            [(1, 1, 1)],             # B11
            [(0, 0, -1), (1, 0, 1)]  # -C00, C01
        ),
        # M6
        (
            [(1, 0, 1), (0, 0, -1)], # A10, -A00
            [(0, 0, 1), (0, 1, 1)],  # B00, B01
            [(1, 1, 1)]              # C11
        ),
        # M7
        (
            [(0, 1, 1), (1, 1, -1)], # A01, -A11
            [(1, 0, 1), (1, 1, 1)],  # B10, B11
            [(0, 0, 1)]              # C00
        ),
    ]
    
    col = 0
    for i_off, j_off, k_off in instances:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for li, lj, coeff in u_coeffs:
                idx_u = (i_off + li) * m + (j_off + lj)
                U[idx_u, col] += coeff
            
            for lj, lk, coeff in v_coeffs:
                idx_v = (j_off + lj) * p + (k_off + lk)
                V[idx_v, col] += coeff
            
            for lk, li, coeff in w_coeffs:
                idx_w = (k_off + lk) * n + (i_off + li)
                W[idx_w, col] += coeff
            
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
