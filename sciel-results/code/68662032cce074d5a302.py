import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct 2x2x2 rank 7 decomposition
    # U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen steps
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices: 00, 01, 10, 11
    # U2 row: i*2+j
    # V2 row: j*2+k
    # W2 row: k*2+i
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1 # A00, A11
    V2[0, 0] = 1; V2[3, 0] = 1 # B00, B11
    W2[0, 0] = 1; W2[1, 0] = 1; W2[2, 0] = 1; W2[3, 0] = 1 # C00, C01, C10, C11 (coeffs: 1, 1, 1, 1)
    # C00 += M1, C01 += M1, C10 += M1, C11 += M1
    # Wait, W2[k*2+i, c]
    # C00: k=0, i=0 -> idx 0
    # C01: k=0, i=1 -> idx 1
    # C10: k=1, i=0 -> idx 2
    # C11: k=1, i=1 -> idx 3
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1 # A10, A11
    V2[0, 1] = 1 # B00
    W2[2, 1] = 1; W2[3, 1] = 1 # C10, C11
    
    # M3
    U2[0, 2] = 1 # A00
    V2[1, 2] = 1; V2[3, 2] = -1 # B01, B11
    W2[1, 2] = 1; W2[3, 2] = 1 # C01, C11
    
    # M4
    U2[3, 3] = 1 # A11
    V2[2, 3] = 1; V2[0, 3] = -1 # B10, B00
    W2[0, 3] = 1; W2[2, 3] = 1 # C00, C10
    
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1 # A00, A01
    V2[3, 4] = 1 # B11
    W2[0, 4] = -1; W2[1, 4] = 1 # C00, C01
    
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1 # A10, A00
    V2[0, 5] = 1; V2[1, 5] = 1 # B00, B01
    W2[3, 5] = 1 # C11
    
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1 # A01, A11
    V2[2, 6] = 1; V2[3, 6] = 1 # B10, B11
    W2[0, 6] = 1; W2[1, 6] = -1 # C00, C01
    
    # 2. Construct 4x4x4 rank 49 decomposition
    # U4[i*4+j, c], V4[j*4+k, c], W4[k*4+i, c]
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Strassen coefficients for 4x4
    # Blocks: 00, 01, 10, 11
    # A00, A01, A10, A11
    # r=0: M1 = (A00+A11)(B00+B11)
    # r=1: M2 = (A10+A11)B00
    # r=2: M3 = A00(B01-B11)
    # r=3: M4 = A11(B10-B00)
    # r=4: M5 = (A00+A01)B11
    # r=5: M6 = (A10-A00)(B00+B01)
    # r=6: M7 = (A01-A11)(B10+B11)
    
    # Coeffs for A and B blocks
    # alpha[r, block]
    alpha = np.zeros((7, 4))
    beta = np.zeros((7, 4))
    # block 0: 00, 1: 01, 2: 10, 3: 11
    alpha[0, 0] = 1; alpha[0, 3] = 1
    beta[0, 0] = 1; beta[0, 3] = 1
    
    alpha[1, 2] = 1; alpha[1, 3] = 1
    beta[1, 0] = 1
    
    alpha[2, 0] = 1
    beta[2, 1] = 1; beta[2, 3] = -1
    
    alpha[3, 3] = 1
    beta[3, 2] = 1; beta[3, 0] = -1
    
    alpha[4, 0] = 1; alpha[4, 1] = 1
    beta[4, 3] = 1
    
    alpha[5, 2] = 1; alpha[5, 0] = -1
    beta[5, 0] = 1; beta[5, 1] = 1
    
    alpha[6, 1] = 1; alpha[6, 3] = -1
    beta[6, 2] = 1; beta[6, 3] = 1
    
    # W coefficients for 4x4
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # W4[k*4+i, c]
    # c = r*7 + s
    # W4 depends on r and s.
    # For a given r, the output blocks are determined.
    # But W4 is for the full 4x4 output.
    # The inner Strassen step s produces the 2x2 result for the block.
    # The outer Strassen step r determines which output block it contributes to.
    
    # Let's map (r, s) to output block (k_block, i_block) and coeff.
    # Actually, W4[k*4+i, r*7+s] = W2[k_local*2+i_local, s] * coeff(r, k_block, i_block)
    # where k_local, i_local are indices within the 2x2 block.
    
    # Output block mapping for r:
    # r=0 (M1): contributes to C00, C01, C10, C11
    # r=1 (M2): contributes to C10, C11
    # r=2 (M3): contributes to C01, C11
    # r=3 (M4): contributes to C00, C10
    # r=4 (M5): contributes to C00, C01
    # r=5 (M6): contributes to C11
    # r=6 (M7): contributes to C00, C01
    
    # Coefficients for each output block (kb, ib) for each r
    # C00: kb=0, ib=0
    # C01: kb=0, ib=1
    # C10: kb=1, ib=0
    # C11: kb=1, ib=1
    
    w_coeffs = np.zeros((7, 4)) # w_coeffs[r, block_idx]
    # block_idx: 0=C00, 1=C01, 2=C10, 3=C11
    w_coeffs[0, 0] = 1; w_coeffs[0, 1] = 1; w_coeffs[0, 2] = 1; w_coeffs[0, 3] = 1
    w_coeffs[1, 2] = 1; w_coeffs[1, 3] = 1
    w_coeffs[2, 1] = 1; w_coeffs[2, 3] = 1
    w_coeffs[3, 0] = 1; w_coeffs[3, 2] = 1
    w_coeffs[4, 0] = -1; w_coeffs[4, 1] = 1
    w_coeffs[5, 3] = 1
    w_coeffs[6, 0] = 1; w_coeffs[6, 1] = -1
    
    # Fill U4, V4, W4
    for r in range(7):
        for s in range(7):
            c = r * 7 + s
            # U4
            for b in range(4):
                if alpha[r, b] != 0:
                    # Block b corresponds to i_block, j_block
                    ib, jb = b // 2, b % 2
                    for i_loc in range(2):
                        for j_loc in range(2):
                            i = ib * 2 + i_loc
                            j = jb * 2 + j_loc
                            u_val = U2[i_loc * 2 + j_loc, s]
                            if u_val != 0:
                                U4[i * 4 + j, c] += alpha[r, b] * u_val
            # V4
            for b in range(4):
                if beta[r, b] != 0:
                    # Block b corresponds to j_block, k_block
                    jb, kb = b // 2, b % 2
                    for j_loc in range(2):
                        for k_loc in range(2):
                            j = jb * 2 + j_loc
                            k = kb * 2 + k_loc
                            v_val = V2[j_loc * 2 + k_loc, s]
                            if v_val != 0:
                                V4[j * 4 + k, c] += beta[r, b] * v_val
            # W4
            for b in range(4):
                if w_coeffs[r, b] != 0:
                    # Block b corresponds to k_block, i_block
                    kb, ib = b // 2, b % 2
                    for k_loc in range(2):
                        for i_loc in range(2):
                            k = kb * 2 + k_loc
                            i = ib * 2 + i_loc
                            w_val = W2[k_loc * 2 + i_loc, s]
                            if w_val != 0:
                                W4[k * 4 + i, c] += w_coeffs[r, b] * w_val

    # 3. Construct 4x5x6 rank 99 decomposition
    # U: 20x99, V: 30x99, W: 24x99
    U = np.zeros((20, 99))
    V = np.zeros((30, 99))
    W = np.zeros((24, 99))
    
    # Block 1: A1 B11 (4x4 x 4x4)
    # A1: rows 0-3, cols 0-3
    # B11: rows 0-3, cols 0-3
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, 0:49] = U4[i * 4 + j, :]
    for j in range(4):
        for k in range(4):
            V[j * 6 + k, 0:49] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, :]
            
    # Block 2: A1 B12' (4x4 x 4x4, B has 2 zero cols)
    # A1: rows 0-3, cols 0-3
    # B12': rows 0-3, cols 4-5 (padded to 4x4, so cols 0-3 in local, mapped to 4-5 in global? No)
    # B12 is 4x2. We pad to 4x4.
    # The multiplication is A1 * B12.
    # B12 global indices: rows 0-3, cols 4-5.
    # In the 4x4x4 block, B is 4x4.
    # We map B12 to the first 2 cols of the local 4x4 B.
    # So V4[j*4+k, c] corresponds to B_local[j, k].
    # We want B_local[j, 0] = B_global[j, 4], B_local[j, 1] = B_global[j, 5].
    # B_local[j, 2] = 0, B_local[j, 3] = 0.
    for j in range(4):
        for k in range(2):
            V[j * 6 + (4 + k), 49:98] = V4[j * 4 + k, :]
        # k=2,3 are 0, so V remains 0 there
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, 49:98] = U4[i * 4 + j, :]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 49:98] = W4[k * 4 + i, :]
            
    # Block 3: A2 B2 (4x1 x 1x6)
    # A2: rows 0-3, col 4
    # B2: row 4, cols 0-5
    # Rank 1
    c = 98
    for i in range(4):
        U[i * 5 + 4, c] = 1
    for k in range(6):
        V[4 * 6 + k, c] = 1
    for k in range(6):
        for i in range(4):
            W[k * 4 + i, c] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
