import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 4, 6
    
    # Rank 42 decomposition
    # Split m=4 into two 2s.
    # C = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
    # Each term is 2x2x6.
    # Split 2x2x6 into three 2x2x2.
    # Each 2x2x2 uses Strassen (rank 7).
    # Total rank = 2 * 3 * 7 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Strassen terms for 2x2x2
    # Local indices:
    # U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00-A10)(B11)
    # M4 = (A00+A11)(B10-B11)
    # M5 = (A00+A10)(B11-B10)
    # M6 = (A01+A11)(B10)
    # M7 = (A00-A10)(B00+B01)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M1 + M2 - M5 - M7
    # C11 = M2 - M1 + M3 + M6
    
    strassen_terms = [
        # (U coeffs, V coeffs, W coeffs)
        # Indices: U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11
        # V[0]=B00, V[1]=B01, V[2]=B10, V[3]=B11
        # W[0]=C00, W[1]=C01, W[2]=C10, W[3]=C11
        
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
        # M3
        ([1, 0, -1, 0], [0, 0, 0, 1], [0, 1, 0, 1]),
        # M4
        ([1, 0, 0, 1], [0, 0, 1, -1], [-1, 1, 0, 0]),
        # M5
        ([1, 0, 1, 0], [0, 0, -1, 1], [1, 0, -1, 0]),
        # M6
        ([0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]),
        # M7
        ([1, 0, -1, 0], [1, 1, 0, 0], [1, 0, -1, 0]),
    ]
    
    col = 0
    # Part 1: A[:, 0:2] @ B[0:2, :]
    # A local indices 0,1,2,3 map to global U indices:
    # A00 -> 0, A01 -> 1, A10 -> 4, A11 -> 5
    a_map_1 = [0, 1, 4, 5]
    
    # B local rows 0,1.
    # B[0, k] -> global V[0*6 + k] = V[k]
    # B[1, k] -> global V[1*6 + k] = V[6+k]
    
    # C local cols 0..5.
    # C[0, k] -> global W[0*2 + k]? No, W[k*n + i].
    # W index = k * n + i = k * 2 + i.
    # C[0, k] -> W[2k]
    # C[1, k] -> W[2k+1]
    
    for b_col_block in range(3): # 0, 1, 2 corresponding to B cols (0,1), (2,3), (4,5)
        # B local col 0 maps to global B col 2*b_col_block
        # B local col 1 maps to global B col 2*b_col_block + 1
        
        # V mapping:
        # Local V[0] (B00) -> Global V[0*6 + 2*b_col_block]
        # Local V[1] (B01) -> Global V[0*6 + 2*b_col_block + 1]
        # Local V[2] (B10) -> Global V[1*6 + 2*b_col_block]
        # Local V[3] (B11) -> Global V[1*6 + 2*b_col_block + 1]
        v_map = [
            0 * 6 + 2 * b_col_block,
            0 * 6 + 2 * b_col_block + 1,
            1 * 6 + 2 * b_col_block,
            1 * 6 + 2 * b_col_block + 1
        ]
        
        # W mapping:
        # Local W[0] (C00) -> Global W[0*2 + 2*b_col_block] = W[2*b_col_block]
        # Local W[1] (C01) -> Global W[0*2 + 2*b_col_block + 1] = W[2*b_col_block + 1]
        # Local W[2] (C10) -> Global W[1*2 + 2*b_col_block] = W[2 + 2*b_col_block]
        # Local W[3] (C11) -> Global W[1*2 + 2*b_col_block + 1] = W[3 + 2*b_col_block]
        w_map = [
            0 * 2 + 2 * b_col_block,
            0 * 2 + 2 * b_col_block + 1,
            1 * 2 + 2 * b_col_block,
            1 * 2 + 2 * b_col_block + 1
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for local_idx, val in enumerate(u_coeffs):
                if val != 0:
                    U[a_map_1[local_idx]][col] += val
            for local_idx, val in enumerate(v_coeffs):
                if val != 0:
                    V[v_map[local_idx]][col] += val
            for local_idx, val in enumerate(w_coeffs):
                if val != 0:
                    W[w_map[local_idx]][col] += val
            col += 1
            
    # Part 2: A[:, 2:4] @ B[2:4, :]
    # A local indices 0,1,2,3 map to global U indices:
    # A02 -> 2, A03 -> 3, A12 -> 6, A13 -> 7
    a_map_2 = [2, 3, 6, 7]
    
    # B local rows 2,3.
    # B[2, k] -> global V[2*6 + k] = V[12+k]
    # B[3, k] -> global V[3*6 + k] = V[18+k]
    
    for b_col_block in range(3):
        v_map = [
            2 * 6 + 2 * b_col_block,
            2 * 6 + 2 * b_col_block + 1,
            3 * 6 + 2 * b_col_block,
            3 * 6 + 2 * b_col_block + 1
        ]
        
        # W mapping is the same as Part 1 because C is the same output
        w_map = [
            0 * 2 + 2 * b_col_block,
            0 * 2 + 2 * b_col_block + 1,
            1 * 2 + 2 * b_col_block,
            1 * 2 + 2 * b_col_block + 1
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for local_idx, val in enumerate(u_coeffs):
                if val != 0:
                    U[a_map_2[local_idx]][col] += val
            for local_idx, val in enumerate(v_coeffs):
                if val != 0:
                    V[v_map[local_idx]][col] += val
            for local_idx, val in enumerate(w_coeffs):
                if val != 0:
                    W[w_map[local_idx]][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
