import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 54 decomposition using Strassen for 2x2 blocks
    # A (2x5) = [A0 (2x2), A1 (2x2), A2 (2x1)]
    # B (5x6) = [B0 (2x6), B1 (2x6), B2 (1x6)]
    # C = A0*B0 + A1*B1 + A2*B2
    
    # B0, B1 are split into three 2x2 blocks each.
    # B0 = [B0_0, B0_1, B0_2] (each 2x2)
    # B1 = [B1_0, B1_1, B1_2] (each 2x2)
    
    # Total Ranks:
    # 6 Strassen blocks * 7 ranks/block = 42
    # 1 Tail block (2x1x6) * 12 ranks = 12
    # Total = 54
    
    rank = 54
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen Coefficients
    # X, Y, Z are 2x2 matrices flattened as [00, 01, 10, 11]
    # M1..M7 correspond to ranks 0..6 within a block
    
    # u_r[i*2+j] is coeff of X_ij in M_r
    u_coeffs = [
        [1, 0, 0, 1], # M1: X00 + X11
        [0, 0, 1, 1], # M2: X10 + X11
        [1, 0, 0, 0], # M3: X00
        [0, 0, 0, 1], # M4: X11
        [1, 0, 1, 0], # M5: X00 + X10
        [-1, 0, 1, 0],# M6: X10 - X00
        [0, 1, 0, -1] # M7: X01 - X11
    ]
    
    # v_r[j*2+k] is coeff of Y_jk in M_r
    v_coeffs = [
        [1, 0, 0, 1], # M1: Y00 + Y11
        [1, 0, 0, 0], # M2: Y00
        [0, -1, 1, 0],# M3: Y10 - Y11
        [0, 1, -1, 0],# M4: Y01 - Y10
        [0, 0, 0, 1], # M5: Y11
        [1, 0, 1, 0], # M6: Y00 + Y10
        [0, 1, 0, 1]  # M7: Y01 + Y11
    ]
    
    # w_ik[r] is coeff of M_r in Z_ik
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    w_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],  # Z00
        [0, 0, 1, 0, 1, 0, 0],   # Z01
        [0, 1, 0, 1, 0, 0, 0],   # Z10
        [1, -1, 1, 0, 0, 1, 0]   # Z11
    ]
    
    # Define the 6 Strassen blocks
    # Each block is defined by:
    # i_map: global i indices for local 0, 1
    # j_map: global j indices for local 0, 1
    # k_map: global k indices for local 0, 1
    
    blocks = [
        # Block 0: A[0:2, 0:2] * B[0:2, 0:2]
        {'i': [0, 1], 'j': [0, 1], 'k': [0, 1]},
        # Block 1: A[0:2, 0:2] * B[0:2, 2:4]
        {'i': [0, 1], 'j': [0, 1], 'k': [2, 3]},
        # Block 2: A[0:2, 0:2] * B[0:2, 4:6]
        {'i': [0, 1], 'j': [0, 1], 'k': [4, 5]},
        # Block 3: A[0:2, 2:4] * B[2:4, 0:2]
        {'i': [0, 1], 'j': [2, 3], 'k': [0, 1]},
        # Block 4: A[0:2, 2:4] * B[2:4, 2:4]
        {'i': [0, 1], 'j': [2, 3], 'k': [2, 3]},
        # Block 5: A[0:2, 2:4] * B[2:4, 4:6]
        {'i': [0, 1], 'j': [2, 3], 'k': [4, 5]},
    ]
    
    current_rank = 0
    for block in blocks:
        i_map = block['i']
        j_map = block['j']
        k_map = block['k']
        
        for r in range(7):
            R = current_rank + r
            u_r = u_coeffs[r]
            v_r = v_coeffs[r]
            
            # Update U
            for li in range(2):
                for lj in range(2):
                    gi = i_map[li]
                    gj = j_map[lj]
                    a_idx = gi * m + gj
                    coeff = u_r[li * 2 + lj]
                    U[a_idx, R] += coeff
                    
            # Update V
            for lj in range(2):
                for lk in range(2):
                    gj = j_map[lj]
                    gk = k_map[lk]
                    b_idx = gj * p + gk
                    coeff = v_r[lj * 2 + lk]
                    V[b_idx, R] += coeff
                    
            # Update W
            for li in range(2):
                for lk in range(2):
                    gi = i_map[li]
                    gk = k_map[lk]
                    c_idx = gk * n + gi
                    # w_coeffs index: li*2+lk corresponds to Z_ik
                    coeff = w_coeffs[li * 2 + lk][r]
                    W[c_idx, R] += coeff
        
        current_rank += 7
        
    # Tail Block: A[0:2, 4:5] * B[4:5, 0:6]
    # A2 is 2x1 (col 4), B2 is 1x6 (row 4)
    # C contribution: C[i, k] += A[i, 4] * B[4, k]
    # Rank 12
    
    for li in range(2):
        gi = i_map[li] # Reuse i_map from last block? No, i_map is [0,1] for all.
        for lk in range(6):
            gk = lk
            R = current_rank + li * 6 + lk
            
            a_idx = gi * m + 4
            b_idx = 4 * p + gk
            c_idx = gk * n + gi
            
            U[a_idx, R] = 1.0
            V[b_idx, R] = 1.0
            W[c_idx, R] = 1.0
            
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
