import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition of 2x2x2
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = A B
    # Indices for A: 0:11, 1:12, 2:21, 3:22
    # Indices for B: 0:11, 1:12, 2:21, 3:22
    # Indices for C: 0:11, 1:12, 2:21, 3:22
    
    # The 7 terms
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B21-B22)
    # M4: A22(B12-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # U vectors (for A part)
    u_strassen = [
        [1, 0, 0, 1], # M1: A11 + A22
        [0, 0, 1, 1], # M2: A21 + A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11 + A12
        [-1, 0, 1, 0], # M6: A21 - A11
        [0, 1, 0, -1]  # M7: A12 - A22
    ]
    
    # V vectors (for B part)
    v_strassen = [
        [1, 0, 0, 1], # M1: B11 + B22
        [1, 0, 0, 0], # M2: B11
        [0, -1, 1, 0], # M3: B21 - B22
        [0, 1, -1, 0], # M4: B12 - B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11 + B12
        [0, 0, 1, 1]  # M7: B21 + B22
    ]
    
    # W vectors (for C part)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M3 + M6
    w_strassen = [
        [1, 0, 0, 1], # M1 contributes to C11 and C22
        [0, 0, 1, -1], # M2 contributes to C21 and -C22
        [0, 1, 0, 1], # M3 contributes to C12 and C22
        [1, 0, 0, 0], # M4 contributes to C11
        [-1, 1, 0, 0], # M5 contributes to -C11 and C12
        [0, 0, 1, 1], # M6 contributes to C21 and C22
        [1, 0, 0, 0]  # M7 contributes to C11
    ]
    
    # Define the 8 blocks
    # Block b: (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    # A is 2x4. B is 4x8. C is 2x8.
    # Local A is 2x2. Local B is 2x2. Local C is 2x2.
    
    blocks = [
        # (a_r, a_c, b_r, b_c, c_c)
        (0, 0, 0, 0, 0), # Block 0: A1, B11L -> C00
        (0, 0, 0, 2, 2), # Block 1: A1, B11R -> C01
        (0, 0, 0, 4, 4), # Block 2: A1, B12L -> C02
        (0, 0, 0, 6, 6), # Block 3: A1, B12R -> C03
        (0, 2, 2, 0, 0), # Block 4: A2, B21L -> C10
        (0, 2, 2, 2, 2), # Block 5: A2, B21R -> C11
        (0, 2, 2, 4, 4), # Block 6: A2, B22L -> C12
        (0, 2, 2, 6, 6), # Block 7: A2, B22R -> C13
    ]
    
    col = 0
    for b_idx in range(8):
        a_r, a_c, b_r, b_c, c_c = blocks[b_idx]
        for k in range(7):
            # u_strassen[k] is length 4.
            # Map local A indices (0,1,2,3) to global A indices.
            # Local (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
            # Global A index for local (i,j) is (a_r + i) * m + (a_c + j)
            # m=4
            
            # v_strassen[k] is length 4.
            # Map local B indices (0,1,2,3) to global B indices.
            # Global B index for local (i,j) is (b_r + i) * p + (b_c + j)
            # p=8
            
            # w_strassen[k] is length 4.
            # Map local C indices (0,1,2,3) to global C indices.
            # Global C index for local (i,j) is (c_c + j) * n + (a_r + i)
            # Wait, W index convention: W[k*n+i, r].
            # Local C (i,j) corresponds to output row i, col j.
            # Global output row is a_r + i (which is just i since a_r=0).
            # Global output col is c_c + j.
            # Global W index = (c_c + j) * n + (a_r + i).
            
            for l in range(4):
                val_u = u_strassen[k][l]
                val_v = v_strassen[k][l]
                val_w = w_strassen[k][l]
                
                if val_u != 0:
                    # l is local A index.
                    # l=0 -> (0,0), l=1 -> (0,1), l=2 -> (1,0), l=3 -> (1,1)
                    li = l // 2
                    lj = l % 2
                    global_a_idx = (a_r + li) * m + (a_c + lj)
                    U[global_a_idx][col] += val_u
                    
                if val_v != 0:
                    li = l // 2
                    lj = l % 2
                    global_b_idx = (b_r + li) * p + (b_c + lj)
                    V[global_b_idx][col] += val_v
                    
                if val_w != 0:
                    li = l // 2
                    lj = l % 2
                    global_c_idx = (c_c + lj) * n + (a_r + li)
                    W[global_c_idx][col] += val_w
                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
