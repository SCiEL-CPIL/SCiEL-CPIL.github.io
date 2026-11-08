import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 84 from 12 Strassen decompositions of 2x2x2
    rank = 84
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    # Strassen terms for 2x2x2
    # Each term is a list of (A_coeff_list, B_coeff_list, C_coeff_list)
    # A/B/C indices are (local_row, local_col)
    # Coefficients are integers
    strassen_terms = [
        # M1: (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
        # M2: (A10+A11)B00 -> C10+=M2, C11-=M2
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (1,0,1), (1,1,-1) ]),
        # M3: A00(B10-B00) -> C01+=M3, C11+=M3
        ([ (0,0,1) ], [ (1,0,1), (0,0,-1) ], [ (0,1,1), (1,1,1) ]),
        # M4: A11(B01-B11) -> C00+=M4, C10+=M4
        ([ (1,1,1) ], [ (0,1,1), (1,1,-1) ], [ (0,0,1), (1,0,1) ]),
        # M5: (A00+A01)B11 -> C00-=M5, C01+=M5
        ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,-1), (0,1,1) ]),
        # M6: (A10-A00)(B00+B01) -> C11+=M6
        ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]),
        # M7: (A01-A11)(B10+B11) -> C00+=M7
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,1) ]),
    ]
    
    col_idx = 0
    # Iterate over the 12 block pairs
    # I: row block of A (0,1)
    # J: col block of A / row block of B (0,1)
    # K: col block of B (0,1,2)
    for I in range(2):
        for J in range(2):
            for K in range(3):
                for A_coeffs, B_coeffs, C_coeffs in strassen_terms:
                    # Add contributions to U (A side)
                    for (lr, lc, coeff) in A_coeffs:
                        # Global row i = 2*I + lr
                        # Global col j = 2*J + lc
                        global_i = 2 * I + lr
                        global_j = 2 * J + lc
                        u_row = global_i * m + global_j
                        U[u_row][col_idx] += coeff
                    
                    # Add contributions to V (B side)
                    for (lr, lc, coeff) in B_coeffs:
                        # Global row j = 2*J + lr
                        # Global col k = 2*K + lc
                        global_j = 2 * J + lr
                        global_k = 2 * K + lc
                        v_row = global_j * p + global_k
                        V[v_row][col_idx] += coeff
                    
                    # Add contributions to W (C side)
                    for (lr, lc, coeff) in C_coeffs:
                        # Global row k = 2*K + lr
                        # Global col i = 2*I + lc
                        global_k = 2 * K + lr
                        global_i = 2 * I + lc
                        # W index is k*n + i
                        w_row = global_k * n + global_i
                        W[w_row][col_idx] += coeff
                    
                    col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
