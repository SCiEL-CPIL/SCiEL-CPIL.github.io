import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28
    rank = 28
    
    # Initialize U, V, W
    # U: (n*m, rank) = (8, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (8, 28)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 matrices
    # X, Y are 2x2. Z = XY.
    # Indices for X: (x_i, x_j) with i,j in {0,1}
    # Indices for Y: (y_j, y_k) with j,k in {0,1}
    # Indices for Z: (z_i, z_k) with i,k in {0,1}
    
    # We define the coefficients for U, V, W for one 2x2x2 multiplication.
    # Each entry is a list of 7 tuples (u_val, v_val, w_val)
    # Actually, it's easier to define the 7 terms directly.
    
    # Term r:
    # U_r is a 4x1 vector (flattened X)
    # V_r is a 4x1 vector (flattened Y)
    # W_r is a 4x1 vector (flattened Z)
    
    # Strassen definitions:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Let's list the (U, V, W) contributions for each M.
    # U indices: 0:X00, 1:X01, 2:X10, 3:X11
    # V indices: 0:Y00, 1:Y01, 2:Y10, 3:Y11
    # W indices: 0:Z00, 1:Z01, 2:Z10, 3:Z11
    
    strassen_terms = [
        # M1
        { 'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 0] },
        # M2
        { 'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 0, 1, 0] },
        # M3
        { 'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 1, 0, 0] },
        # M4
        { 'U': [0, 0, 0, 1], 'V': [0, 0, 1, -1], 'W': [1, 0, 1, 0] },
        # M5
        { 'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 1, 0, 0] },
        # M6
        { 'U': [0, 0, 1, -1], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1] },
        # M7
        { 'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 1] },
    ]
    
    # We have 4 sub-problems:
    # 1. A1 * B_topL (A cols 0,1; B rows 0,1 cols 0,1)
    # 2. A2 * B_botL (A cols 2,3; B rows 2,3 cols 0,1)
    # 3. A1 * B_topR (A cols 0,1; B rows 0,1 cols 2,3)
    # 4. A2 * B_botR (A cols 2,3; B rows 2,3 cols 2,3)
    
    col = 0
    for sub_idx in range(4):
        # Determine offsets
        if sub_idx == 0 or sub_idx == 2:
            a_offset = 0 # A1
        else:
            a_offset = 2 # A2
            
        if sub_idx == 0 or sub_idx == 1:
            b_row_offset = 0 # Top
            b_col_offset = 0 # Left
        else:
            b_row_offset = 2 # Bot
            b_col_offset = 2 # Right
            
        # Iterate through 7 Strassen terms
        for term in strassen_terms:
            # Update U
            # U index for A: i*m + j
            # i in 0,1; j in 0,1 (local)
            for i in range(2):
                for j in range(2):
                    u_val = term['U'][i*2 + j]
                    if u_val != 0:
                        global_j = a_offset + j
                        u_idx = i * m + global_j
                        U[u_idx][col] += u_val
                        
            # Update V
            # V index for B: j*p + k
            # j in 0,1 (local row); k in 0,1 (local col)
            for j in range(2):
                for k in range(2):
                    v_val = term['V'][j*2 + k]
                    if v_val != 0:
                        global_j = b_row_offset + j
                        global_k = b_col_offset + k
                        v_idx = global_j * p + global_k
                        V[v_idx][col] += v_val
                        
            # Update W
            # W index for C: k*n + i
            # k in 0,1 (local col); i in 0,1
            for k in range(2):
                for i in range(2):
                    w_val = term['W'][i*2 + k] # Note: W list is [Z00, Z01, Z10, Z11] -> i*2+k
                    if w_val != 0:
                        global_k = b_col_offset + k
                        w_idx = global_k * n + i
                        W[w_idx][col] += w_val
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
