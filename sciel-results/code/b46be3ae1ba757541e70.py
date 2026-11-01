"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin

    n, m, p = DIMENSIONS
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16

    # Strassen factors for 2x2x2 multiplication.
    # Inputs: A=[a00,a01,a10,a11], B=[b00,b01,b10,b11]
    # Outputs: C=[c00,c01,c10,c11]
    u_strassen = np.array([
        [1, 0, 0, 1],   # M1: a00+a11
        [0, 0, 1, 1],   # M2: a10+a11
        [1, 0, 0, 0],   # M3: a00
        [0, 0, 0, 1],   # M4: a11
        [1, 1, 0, 0],   # M5: a00+a01
        [-1, 0, 1, 0],  # M6: a10-a00
        [0, 1, 0, -1]   # M7: a01-a11
    ], dtype=int)

    v_strassen = np.array([
        [1, 0, 0, 1],   # M1: b00+b11
        [1, 0, 0, 0],   # M2: b00
        [0, 0, 1, -1],  # M3: b10-b11
        [0, 1, -1, 0],  # M4: b01-b10
        [0, 0, 0, 1],   # M5: b11
        [1, 0, 1, 0],   # M6: b00+b10
        [0, 1, 0, 1]    # M7: b01+b11
    ], dtype=int)

    w_strassen = np.array([
        [1, 0, 0, 1],   # M1: c00+c11
        [0, 0, 1, -1],  # M2: c10-c11
        [0, 1, 0, 1],   # M3: c01+c11
        [1, 0, 1, 0],   # M4: c00+c10
        [-1, 1, 0, 0],  # M5: -c00+c01
        [0, 0, 0, 1],   # M6: c11
        [1, 0, 0, 0]    # M7: c00
    ], dtype=int)

    # Build rank-56 Strassen decomposition
    # We have 4 blocks of 2x2 matrices in the 4x8 product dimension?
    # No, A is 2x4, B is 4x8.
    # We can view B as four 4x2 blocks? Or two 4x4 blocks?
    # Let's use the 4 blocks of 2x2 multiplications derived from splitting A and B.
    # A = [A1 A2] (2x2 each)
    # B = [B11 B12; B21 B22] (2x2 each)
    # C = [A1 B11 + A2 B21, A1 B12 + A2 B22]
    # This requires 4 products of 2x2 matrices.
    # Each 2x2 product has rank 7.
    # Total rank 28 for 2x4x4.
    # Since B is 4x8, we have two such 2x4x4 multiplications (for the two halves of B).
    # Total rank 56.

    U_global = np.zeros((A_dim, 56), dtype=int)
    V_global = np.zeros((B_dim, 56), dtype=int)
    W_global = np.zeros((C_dim, 56), dtype=int)

    col_offset = 0

    # The 4 blocks correspond to (A_part, B_part) combinations
    # A_part 0: cols 0,1 of A (indices 0,1,4,5 in A_flat? No, A is row-major 2x4)
    # A indices:
    # 0 1 2 3
    # 4 5 6 7
    # A_L (cols 0,1): 0,1,4,5 -> local 0,1,2,3
    # A_R (cols 2,3): 2,3,6,7 -> local 0,1,2,3
    
    # B is 4x8.
    # B_T (rows 0,1): 0..7
    # B_B (rows 2,3): 8..15
    # For the first half of C (cols 0,3 of B):
    # B11 (rows 0,1, cols 0,1): 0,1,4,5
    # B12 (rows 0,1, cols 2,3): 2,3,6,7
    # B21 (rows 2,3, cols 0,1): 8,9,12,13
    # B22 (rows 2,3, cols 2,3): 10,11,14,15
    
    # For the second half of C (cols 4,7 of B):
    # B11' (rows 0,1, cols 4,5): 4,5,8,9 -- Wait, B is row major.
    # B indices:
    # 0 1 2 3 4 5 6 7
    # 8 9 10 11 12 13 14 15
    # 16 17 18 19 20 21 22 23
    # 24 25 26 27 28 29 30 31
    
    # B11 (r0,1 c0,1): 0,1,8,9
    # B12 (r0,1 c2,3): 2,3,10,11
    # B21 (r2,3 c0,1): 16,17,24,25
    # B22 (r2,3 c2,3): 18,19,26,27
    
    # B11' (r0,1 c4,5): 4,5,12,13
    # B12' (r0,1 c6,7): 6,7,14,15
    # B21' (r2,3 c4,5): 20,21,28,29
    # B22' (r2,3 c6,7): 22,23,30,31

    # C is 2x8.
    # C indices:
    # 0 1 2 3 4 5 6 7
    # 8 9 10 11 12 13 14 15
    # C1 (cols 0,1): 0,1,8,9
    # C2 (cols 2,3): 2,3,10,11
    # C3 (cols 4,5): 4,5,12,13
    # C4 (cols 6,7): 6,7,14,15

    # Block 1: A_L * B11 -> C1
    # Block 2: A_R * B21 -> C1
    # Block 3: A_L * B12 -> C2
    # Block 4: A_R * B22 -> C2
    
    # Block 5: A_L * B11' -> C3
    # Block 6: A_R * B21' -> C3
    # Block 7: A_L * B12' -> C4
    # Block 8: A_R * B22' -> C4

    a_l_idx = [0, 1, 4, 5]
    a_r_idx = [2, 3, 6, 7]
    
    b_blocks = [
        [0, 1, 8, 9],   # B11
        [16, 17, 24, 25], # B21
        [2, 3, 10, 11], # B12
        [18, 19, 26, 27], # B22
        [4, 5, 12, 13], # B11'
        [20, 21, 28, 29], # B21'
        [6, 7, 14, 15], # B12'
        [22, 23, 30, 31]  # B22'
    ]
    
    c_blocks = [
        [0, 1, 8, 9],   # C1
        [0, 1, 8, 9],   # C1
        [2, 3, 10, 11], # C2
        [2, 3, 10, 11], # C2
        [4, 5, 12, 13], # C3
        [4, 5, 12, 13], # C3
        [6, 7, 14, 15], # C4
        [6, 7, 14, 15]  # C4
    ]
    
    a_blocks = [
        a_l_idx, # A_L
        a_r_idx, # A_R
        a_l_idx, # A_L
        a_r_idx, # A_R
        a_l_idx, # A_L
        a_r_idx, # A_R
        a_l_idx, # A_L
        a_r_idx  # A_R
    ]

    for b_idx, c_idx, a_idx in zip(b_blocks, c_blocks, a_blocks):
        for s in range(7):
            col = col_offset + s
            for local_i, global_i in enumerate(a_idx):
                U_global[global_i, col] = u_strassen[s, local_i]
            for local_j, global_j in enumerate(b_idx):
                V_global[global_j, col] = v_strassen[s, local_j]
            for local_k, global_k in enumerate(c_idx):
                W_global[global_k, col] = w_strassen[s, local_k]
        col_offset += 7

    # Verify the decomposition
    T = np.zeros((A_dim, B_dim, C_dim), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1

    T_rec = np.einsum('ar,br,cr->abc', U_global.astype(float),
                       V_global.astype(float), W_global.astype(float))

    if not np.allclose(T, T_rec, atol=1e-10):
        # Fallback: schoolbook rank-64
        rank = 64
        U_s = np.zeros((A_dim, rank), dtype=int)
        V_s = np.zeros((B_dim, rank), dtype=int)
        W_s = np.zeros((C_dim, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m + j, col] = 1
                    V_s[j*p + k, col] = 1
                    W_s[k*n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
# EVOLVE-BLOCK-END
