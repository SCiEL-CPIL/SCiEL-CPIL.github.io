import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    def get_U_index(row, col):
        return row * m + col
    
    def get_V_index(row, col):
        return row * p + col
    
    def get_W_index(row, col):
        return col * n + row
    
    def a_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def b_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    def c_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
    
    # Define the 7 Strassen terms for 2x2 blocks
    # Each term: (A_block_pattern, B_block_pattern, C_contribution_pattern)
    # A_block_pattern: dict mapping (block_row, block_col) -> coeff
    # B_block_pattern: dict mapping (block_row, block_col) -> coeff  
    # C_contribution: dict mapping (block_row, block_col) -> coeff
    
    strassen_terms = [
        # M1 = (A11 + A22)(B11 + B22) -> C11 += M1, C22 += M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A21 + A22)B11 -> C21 += M2, C22 -= M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A11(B12 - B22) -> C12 += M3, C22 += M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A22(B21 - B11) -> C11 += M4, C21 += M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A11 + A12)B22 -> C11 -= M5, C12 += M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A21 - A11)(B11 + B12) -> C21 += M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A12 - A22)(B21 + B22) -> C11 += M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # For each of the 4 block positions (bi, bj), apply the 7 Strassen terms
    # to the 2x2 sub-blocks at that position
    for bi in range(2):
        for bj in range(2):
            for t in range(7):
                A_pattern, B_pattern, C_pattern = strassen_terms[t]
                rank_idx = (bi * 2 + bj) * 7 + t
                
                # Build U contributions from A_pattern
                for (ar, ac), a_coeff in A_pattern.items():
                    u_row, u_col = a_elem(bi, bj, ar, ac)
                    U[get_U_index(u_row, u_col)][rank_idx] += a_coeff
                
                # Build V contributions from B_pattern
                for (br, bc), b_coeff in B_pattern.items():
                    v_row, v_col = b_elem(bi, bj, br, bc)
                    V[get_V_index(v_row, v_col)][rank_idx] += b_coeff
                
                # Build W contributions from C_pattern
                for (cr, cc), c_coeff in C_pattern.items():
                    w_row, w_col = c_elem(bi, bj, cr, cc)
                    W[get_W_index(w_row, w_col)][rank_idx] += c_coeff
    
    return {"U": U, "V": V, "W": W}
