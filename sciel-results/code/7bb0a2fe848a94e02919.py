import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to flatten indices
    def idx_A(i, j):
        return i * m + j
        
    def idx_B(i, j):
        return i * p + j
        
    def idx_C(i, j):
        return i * n + j
        
    # Define the 7 outer Strassen A-combinations (2x2 matrices of coeffs for A)
    # S_A[k] is a 2x2 matrix, each entry is a dict or vector of coeffs for A[i,j]
    # We'll use lists of (coeff, i, j)
    
    # A blocks:
    # A11: (0,0), (0,1), (1,0), (1,1)
    # A12: (0,2), (0,3), (1,2), (1,3)
    # A21: (2,0), (2,1), (3,0), (3,1)
    # A22: (2,2), (2,3), (3,2), (3,3)
    
    # We define S_A[k][r][c] as a list of (coeff, A_idx)
    
    def get_A_block(r, c):
        # r, c in {0, 1} for block row/col
        # Returns list of (1, global_idx) for the 4 entries
        base_i = r * 2
        base_j = c * 2
        return [
            (1, idx_A(base_i, base_j)),
            (1, idx_A(base_i, base_j + 1)),
            (1, idx_A(base_i + 1, base_j)),
            (1, idx_A(base_i + 1, base_j + 1))
        ]
        
    def add_block_coeffs(target, coeff, block_r, block_c):
        # Add coeff * A_block[block_r, block_c] to target
        entries = get_A_block(block_r, block_c)
        for c, idx in entries:
            # target is a dict {idx: coeff} or similar
            # Let's use a dict for accumulation
            if idx in target:
                target[idx] += coeff * c
            else:
                target[idx] = coeff * c
                
    S_A = []
    for k in range(7):
        S_A_k = [[{} for _ in range(2)] for _ in range(2)]
        
        if k == 0: # M1: A11 + A22
            add_block_coeffs(S_A_k[0][0], 1, 0, 0)
            add_block_coeffs(S_A_k[0][1], 1, 0, 0) # Wait, A11 is 2x2.
            # My get_A_block returns 4 entries.
            # S_A_k[0][0] corresponds to entry (0,0) of the 2x2 matrix S_A.
            # So S_A_k[0][0] should contain the coeffs for A11[0,0]?
            # No, S_A is a 2x2 matrix of linear forms.
            # S_A[0][0] is the linear form for the top-left element of the 2x2 result.
            
            # Let's re-evaluate.
            # S_A[k] is a 2x2 matrix.
            # S_A[k][r][c] is a linear combination of A entries.
            
            # M1: (A11 + A22)
            # S_A[0][0] = A11[0,0] + A22[0,0]
            # S_A[0][1] = A11[0,1] + A22[0,1]
            # S_A[1][0] = A11[1,0] + A22[1,0]
            # S_A[1][1] = A11[1,1] + A22[1,1]
            
            # This is getting tedious to do manually.
            # Let's just define the linear forms explicitly.
            
            pass
            
    # Let's hardcode the 7 A-combinations as 2x2 matrices of coefficients for the 16 A entries.
    # Entry (r,c) of S_A[k] is a vector of length 16.
    
    def make_zero_2x2():
        return [[np.zeros(16) for _ in range(2)] for _ in range(2)]
        
    def set_entry(mat, r, c, a_idx, val):
        mat[r][c][a_idx] += val
        
    S_A = [make_zero_2x2() for _ in range(7)]
    
    # M1: A11 + A22
    # A11: 0,1,4,5 (indices 0,1,4,5? No. 0,1; 4,5. Yes.)
    # A22: 8,9,12,13? No.
    # A indices:
    # 0 1 2 3
    # 4 5 6 7
    # 8 9 10 11
    # 12 13 14 15
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # M1: A11 + A22
    for r in range(2):
        for c in range(2):
            a11_idx = [0,1,4,5][r*2+c]
            a22_idx = [10,11,14,15][r*2+c]
            set_entry(S_A[0], r, c, a11_idx, 1)
            set_entry(S_A[0], r, c, a22_idx, 1)
            
    # M2: A21 + A22
    for r in range(2):
        for c in range(2):
            a21_idx = [8,9,12,13][r*2+c]
            a22_idx = [10,11,14,15][r*2+c]
            set_entry(S_A[1], r, c, a21_idx, 1)
            set_entry(S_A[1], r, c, a22_idx, 1)
            
    # M3: A11
    for r in range(2):
        for c in range(2):
            a11_idx = [0,1,4,5][r*2+c]
            set_entry(S_A[2], r, c, a11_idx, 1)
            
    # M4: A22
    for r in range(2):
        for c in range(2):
            a22_idx = [10,11,14,15][r*2+c]
            set_entry(S_A[3], r, c, a22_idx, 1)
            
    # M5: A11 + A12
    for r in range(2):
        for c in range(2):
            a11_idx = [0,1,4,5][r*2+c]
            a12_idx = [2,3,6,7][r*2+c]
            set_entry(S_A[4], r, c, a11_idx, 1)
            set_entry(S_A[4], r, c, a12_idx, 1)
            
    # M6: A21 - A11
    for r in range(2):
        for c in range(2):
            a21_idx = [8,9,12,13][r*2+c]
            a11_idx = [0,1,4,5][r*2+c]
            set_entry(S_A[5], r, c, a21_idx, 1)
            set_entry(S_A[5], r, c, a11_idx, -1)
            
    # M7: A12 - A22
    for r in range(2):
        for c in range(2):
            a12_idx = [2,3,6,7][r*2+c]
            a22_idx = [10,11,14,15][r*2+c]
            set_entry(S_A[6], r, c, a12_idx, 1)
            set_entry(S_A[6], r, c, a22_idx, -1)
            
    # Now B-combinations.
    # B is 4x8. Indices 0..31.
    # B11: rows 0,1; cols 0,1,2,3 -> 0,1,2,3; 8,9,10,11
    # B12: rows 0,1; cols 4,5,6,7 -> 4,5,6,7; 12,13,14,15
    # B21: rows 2,3; cols 0,1,2,3 -> 16,17,18,19; 24,25,26,27
    # B22: rows 2,3; cols 4,5,6,7 -> 20,21,22,23; 28,29,30,31
    
    def make_zero_2x4():
        return [[np.zeros(32) for _ in range(4)] for _ in range(2)]
        
    def set_entry_B(mat, r, c, b_idx, val):
        mat[r][c][b_idx] += val
        
    S_B = [make_zero_2x4() for _ in range(7)]
    
    # B11 indices: [0,1,2,3, 8,9,10,11]
    # B12 indices: [4,5,6,7, 12,13,14,15]
    # B21 indices: [16,17,18,19, 24,25,26,27]
    # B22 indices: [20,21,22,23, 28,29,30,31]
    
    B_blocks = {
        0: [0,1,2,3, 8,9,10,11],
        1: [4,5,6,7, 12,13,14,15],
        2: [16,17,18,19, 24,25,26,27],
        3: [20,21,22,23, 28,29,30,31]
    }
    
    # M1: B11 + B22
    for r in range(2):
        for c in range(4):
            b11_idx = B_blocks[0][r*4+c]
            b22_idx = B_blocks[3][r*4+c]
            set_entry_B(S_B[0], r, c, b11_idx, 1)
            set_entry_B(S_B[0], r, c, b22_idx, 1)
            
    # M2: B11
    for r in range(2):
        for c in range(4):
            b11_idx = B_blocks[0][r*4+c]
            set_entry_B(S_B[1], r, c, b11_idx, 1)
            
    # M3: B22 - B12
    for r in range(2):
        for c in range(4):
            b22_idx = B_blocks[3][r*4+c]
            b12_idx = B_blocks[1][r*4+c]
            set_entry_B(S_B[2], r, c, b22_idx, 1)
            set_entry_B(S_B[2], r, c, b12_idx, -1)
            
    # M4: B11 - B21
    for r in range(2):
        for c in range(4):
            b11_idx = B_blocks[0][r*4+c]
            b21_idx = B_blocks[2][r*4+c]
            set_entry_B(S_B[3], r, c, b11_idx, 1)
            set_entry_B(S_B[3], r, c, b21_idx, -1)
            
    # M5: B22
    for r in range(2):
        for c in range(4):
            b22_idx = B_blocks[3][r*4+c]
            set_entry_B(S_B[4], r, c, b22_idx, 1)
            
    # M6: B11 + B12
    for r in range(2):
        for c in range(4):
            b11_idx = B_blocks[0][r*4+c]
            b12_idx = B_blocks[1][r*4+c]
            set_entry_B(S_B[5], r, c, b11_idx, 1)
            set_entry_B(S_B[5], r, c, b12_idx, 1)
            
    # M7: B21 + B22
    for r in range(2):
        for c in range(4):
            b21_idx = B_blocks[2][r*4+c]
            b22_idx = B_blocks[3][r*4+c]
            set_entry_B(S_B[6], r, c, b21_idx, 1)
            set_entry_B(S_B[6], r, c, b22_idx, 1)
            
    # Now, define the 14 inner Strassen products for M_{2,2,4}.
    # For each k in 0..6 (outer), and j in 0..13 (inner):
    # We need U_col, V_col, W_col.
    
    # Inner Strassen for X (2x2) and Y_part (2x2).
    # Products:
    # L1: (X11+X22)(Y11+Y22)
    # L2: (X21+X22)Y11
    # L3: X11(Y22-Y12)
    # L4: X22(Y11-Y21)
    # L5: (X11+X12)Y22
    # L6: (X21-X11)(Y11+Y12)
    # L7: (X12-X22)(Y21+Y22)
    
    # Reconstruction for Z (2x2):
    # Z11 = L1 - L4 + L5 - L7
    # Z12 = L3 + L7
    # Z21 = L2 + L4
    # Z22 = L1 - L3 + L6
    
    # We have two sets: L (for Y_L) and R (for Y_R).
    # Y_L corresponds to cols 0-1 of Y_part? No, Y_part is 2x2.
    # In M_{2,2,4}, Y is 2x4. Y_L is cols 0-1, Y_R is cols 2-3.
    
    # For each outer k:
    # S_A[k] is 2x2. S_B[k] is 2x4.
    # We treat S_B[k] as two 2x2 matrices: S_B_L = S_B[k][:, 0:2], S_B_R = S_B[k][:, 2:4].
    
    # For each inner j (0..13):
    # If j < 7, it's an L-product. Y_part = S_B_L.
    # If j >= 7, it's an R-product. Y_part = S_B_R.
    
    # Define linear forms for X (2x2) for the 7 Strassen products.
    # X indices: 0,1,2,3 (row-major)
    # X11=0, X12=1, X21=2, X22=3
    
    X_forms = [
        [1, 0, 0, 1], # L1: X11+X22
        [0, 0, 1, 1], # L2: X21+X22
        [1, 0, 0, 0], # L3: X11
        [0, 0, 0, 1], # L4: X22
        [1, 1, 0, 0], # L5: X11+X12
        [-1, 0, 1, 0],# L6: X21-X11
        [0, 1, 0, -1] # L7: X12-X22
    ]
    
    # Define linear forms for Y (2x2) for the 7 Strassen products.
    # Y indices: 0,1,2,3
    # Y11=0, Y12=1, Y21=2, Y22=3
    
    Y_forms = [
        [1, 0, 0, 1], # L1: Y11+Y22
        [1, 0, 0, 0], # L2: Y11
        [0, -1, 0, 1],# L3: Y22-Y12
        [1, 0, -1, 0],# L4: Y11-Y21
        [0, 0, 0, 1], # L5: Y22
        [1, 1, 0, 0], # L6: Y11+Y12
        [0, 0, 1, 1]  # L7: Y21+Y22
    ]
    
    # Reconstruction coefficients for Z (2x2) from L1..L7.
    # Z11: L1 - L4 + L5 - L7 -> [1, 0, 0, -1, 1, 0, -1]
    # Z12: L3 + L7 -> [0, 0, 1, 0, 0, 0, 1]
    # Z21: L2 + L4 -> [0, 1, 0, 1, 0, 0, 0]
    # Z22: L1 - L3 + L6 -> [1, 0, -1, 0, 0, 1, 0]
    
    Z_recon = [
        [1, 0, 0, -1, 1, 0, -1], # Z11
        [0, 0, 1, 0, 0, 0, 1],   # Z12
        [0, 1, 0, 1, 0, 0, 0],   # Z21
        [1, 0, -1, 0, 0, 1, 0]   # Z22
    ]
    
    # Outer Strassen reconstruction for C (4x8) from M1..M7 (each 2x4).
    # C11 (0:2, 0:4): M1 - M4 + M5 - M7
    # C12 (0:2, 4:8): M3 + M7
    # C21 (2:4, 0:4): M2 + M4
    # C22 (2:4, 4:8): M1 - M3 + M6
    
    # C_recon[block_row, block_col][local_r, local_c] = coeffs for M1..M7
    # local_r, local_c in 0..1, 0..3
    
    C_recon = [
        [ # Block 0,0 (C11)
            [ [1,0,0,-1,1,0,-1] for _ in range(4) ], # r=0
            [ [1,0,0,-1,1,0,-1] for _ in range(4) ]  # r=1
        ],
        [ # Block 0,1 (C12)
            [ [0,0,1,0,0,0,1] for _ in range(4) ],
            [ [0,0,1,0,0,0,1] for _ in range(4) ]
        ],
        [ # Block 1,0 (C21)
            [ [0,1,0,1,0,0,0] for _ in range(4) ],
            [ [0,1,0,1,0,0,0] for _ in range(4) ]
        ],
        [ # Block 1,1 (C22)
            [ [1,0,-1,0,0,1,0] for _ in range(4) ],
            [ [1,0,-1,0,0,1,0] for _ in range(4) ]
        ]
    ]
    
    # Build U, V, W
    col = 0
    for k in range(7):
        for j in range(14):
            is_L = j < 7
            j_inner = j % 7
            
            # U column
            # U[:, col] = sum_{r,c} X_forms[j_inner][r*2+c] * S_A[k][r][c]
            U_col = np.zeros(16)
            for r in range(2):
                for c in range(2):
                    coeff = X_forms[j_inner][r*2+c]
                    if coeff != 0:
                        U_col += coeff * S_A[k][r][c]
            U[:, col] = U_col
            
            # V column
            # V[:, col] = sum_{r,c} Y_forms[j_inner][r*2+c] * S_B[k][r][c_part]
            # c_part is 0..1 for L, 2..3 for R
            V_col = np.zeros(32)
            for r in range(2):
                for c in range(2):
                    coeff = Y_forms[j_inner][r*2+c]
                    if coeff != 0:
                        if is_L:
                            V_col += coeff * S_B[k][r][c]
                        else:
                            V_col += coeff * S_B[k][r][c+2]
            V[:, col] = V_col
            
            # W column
            # For each output entry (i, j_out) in 4x8:
            # Determine block (br, bc) and local (lr, lc)
            # br = i // 2, bc = j_out // 4
            # lr = i % 2, lc = j_out % 4
            
            # The contribution from this (k, j) to W[i, j_out] is:
            # C_recon[br, bc][lr, lc][k] * Z_recon[lr, lc? No, Z is 2x2, but our local output is 2x4?
            
            # Wait. Z_recon is for 2x2 output.
            # Our inner product is M_{2,2,2} (X * Y_part).
            # So Z is 2x2.
            # The local output index in Z is (lr, lc_inner) where lc_inner is 0 or 1.
            # But our global local col lc is 0..3.
            # If is_L, lc_inner = lc (0 or 1).
            # If is_R, lc_inner = lc - 2 (0 or 1).
            
            W_col = np.zeros(32)
            for i in range(4):
                for j_out in range(8):
                    br = i // 2
                    bc = j_out // 4
                    lr = i % 2
                    lc = j_out % 4
                    
                    if is_L:
                        if lc > 1: continue
                        lc_inner = lc
                    else:
                        if lc < 2: continue
                        lc_inner = lc - 2
                    
                    outer_coeff = C_recon[br][bc][lr][lc][k]
                    inner_coeff = Z_recon[lr][lc_inner][j_inner]
                    
                    val = outer_coeff * inner_coeff
                    if val != 0:
                        W_col[idx_C(i, j_out)] = val
                        
            W[:, col] = W_col
            col += 1
            
    return {"U": U, "V": V, "W": W}
