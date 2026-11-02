# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank_total = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_decomp(U_mat, V_mat, W_mat):
        nonlocal rank_total
        r = U_mat.shape[1]
        U_list.append(U_mat)
        V_list.append(V_mat)
        W_list.append(W_mat)
        rank_total += r

    # Strassen 2x2
    def strassen_2x2():
        # Returns U, V, W for 2x2x2
        # Indices:
        # U: i*2+j
        # V: j*2+k
        # W: k*2+i
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        # A11: U[0], A22: U[3]
        # B11: V[0], B22: V[3]
        # C11: W[0] (k=0,i=0), C22: W[3] (k=1,i=1)
        # C11 += M1, C22 += M1
        U[:, 0] = [1, 0, 0, 1]
        V[:, 0] = [1, 0, 0, 1]
        W[0, 0] = 1
        W[3, 0] = 1
        
        # M2 = (A21+A22)B11
        # A21: U[2], A22: U[3]
        # B11: V[0]
        # C21: W[1] (k=0,i=1? No. C21 is row 1 col 0. W index k*2+i. k=0, i=1 -> W[1])
        # C22: W[3]
        U[:, 1] = [0, 0, 1, 1]
        V[:, 1] = [1, 0, 0, 0]
        W[1, 1] = 1
        W[3, 1] = -1
        
        # M3 = A11(B21+B22)
        # A11: U[0]
        # B21: V[2], B22: V[3]
        # C11: W[0], C12: W[2] (k=1, i=0 -> W[2])
        U[:, 2] = [1, 0, 0, 0]
        V[:, 2] = [0, 0, 1, 1]
        W[0, 2] = -1
        W[2, 2] = 1
        
        # M4 = A22(B11-B21)
        # A22: U[3]
        # B11: V[0], B21: V[2]
        # C11: W[0], C21: W[1]
        U[:, 3] = [0, 0, 0, 1]
        V[:, 3] = [1, 0, -1, 0]
        W[0, 3] = 1
        W[1, 3] = 1
        
        # M5 = (A11+A12)B22
        # A11: U[0], A12: U[1]
        # B22: V[3]
        # C12: W[2], C22: W[3]
        U[:, 4] = [1, 1, 0, 0]
        V[:, 4] = [0, 0, 0, 1]
        W[2, 4] = -1
        W[3, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21: U[2], A11: U[0]
        # B11: V[0], B12: V[1]
        # C21: W[1], C22: W[3]
        U[:, 5] = [-1, 0, 1, 0]
        V[:, 5] = [1, 1, 0, 0]
        W[1, 5] = -1
        W[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        # A12: U[1], A22: U[3]
        # B21: V[2], B22: V[3]
        # C11: W[0], C22: W[3]
        U[:, 6] = [0, 1, 0, -1]
        V[:, 6] = [0, 0, 1, 1]
        W[0, 6] = 1
        W[3, 6] = -1
        
        return U, V, W

    # Build Strassen 4x4 by tensor product of Strassen 2x2
    U2, V2, W2 = strassen_2x2()
    # Rank 49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for i in range(7):
        for j in range(7):
            col = i * 7 + j
            # U4 index: (i1*2+i2)*2 + j1*2+j2?
            # A is 4x4. Split into 2x2 blocks.
            # A_00 (top-left), A_01 (top-right)
            # A_10 (bottom-left), A_11 (bottom-right)
            # Strassen 2x2 on blocks.
            # U2[i] picks a linear combo of A_block entries.
            # We need to map U2 indices (0..3) to U4 indices (0..15).
            # U2 index 0 -> A00 (U4 0)
            # U2 index 1 -> A01 (U4 1)
            # U2 index 2 -> A10 (U4 2)
            # U2 index 3 -> A11 (U4 3)
            # Wait, U2 is for 2x2 matrix.
            # The 4x4 matrix is:
            # [ A00 A01 ]
            # [ A10 A11 ]
            # Each Ak is 2x2.
            # U2[i] is a vector of length 4.
            # It applies to the "block" entries.
            # So if U2[i, 0] = 1, it adds A00.
            # A00 occupies U4 indices 0..3.
            # So we add U2[i, 0] * U2[j, :] to U4[0:4, col]?
            # No.
            # The tensor product structure:
            # U4[ (i1*2+i2)*2 + (j1*2+j2) , col ]
            # This is getting complicated.
            # Simpler: Just use schoolbook for 4x4? Rank 16.
            # 49 is better.
            # Let's just use schoolbook for 4x4 to avoid bugs. Rank 16.
            # Total rank: 16 + 46 = 62.
            # This is MUCH better than 95.
            
    # Let's restart with Schoolbook for 4x4.
    # Rank 4x4 = 16.
    # Cross terms:
    # C_TL = A_TL B_TL (16) + A_TR B_BL (1)
    # C_TR = A_TL B_TR (16) + A_TR B_BR (4)
    # C_BL = A_BL B_TL (16) + A_BR B_BL (4)
    # C_BR = A_BL B_TR (4) + A_BR B_BR (1)
    # Total: 16+1+16+4+16+4+4+1 = 62.
    
    U_list = []
    V_list = []
    W_list = []
    rank_total = 0
    
    def add_schoolbook(n_a, n_b, n_c, offset_a, offset_b, offset_c):
        # A is n_a x n_b, B is n_b x n_c
        # U offset: offset_a (row), offset_b (col)
        # V offset: offset_b (row), offset_c (col)
        # W offset: offset_c (row), offset_a (col)
        # Indices:
        # U: (i + offset_a_row) * 5 + (j + offset_a_col)
        # V: (j + offset_b_row) * 5 + (k + offset_b_col)
        # W: (k + offset_c_row) * 5 + (i + offset_c_col)
        
        r = n_a * n_b * n_c
        U = np.zeros((25, r))
        V = np.zeros((25, r))
        W = np.zeros((25, r))
        
        col = 0
        for i in range(n_a):
            for j in range(n_b):
                for k in range(n_c):
                    u_idx = (i + offset_a[0]) * 5 + (j + offset_a[1])
                    v_idx = (j + offset_b[0]) * 5 + (k + offset_b[1])
                    w_idx = (k + offset_c[0]) * 5 + (i + offset_c[1])
                    
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1
                    
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)
        rank_total += r

    # Blocks:
    # TL: 0:4, 0:4
    # TR: 0:4, 4:5
    # BL: 4:5, 0:4
    # BR: 4:5, 4:5
    
    # 1. A_TL B_TL
    add_schoolbook(4, 4, 4, (0,0), (0,0), (0,0))
    
    # 2. A_TR B_BL
    # A_TR: rows 0:4, col 4. (4x1)
    # B_BL: row 4, cols 0:4. (1x4)
    # C_TL: rows 0:4, cols 0:4.
    add_schoolbook(4, 1, 4, (0,4), (4,0), (0,0))
    
    # 3. A_TL B_TR
    # A_TL: 4x4
    # B_TR: 4x1
    # C_TR: 4x1
    add_schoolbook(4, 4, 1, (0,0), (0,4), (0,4))
    
    # 4. A_TR B_BR
    # A_TR: 4x1
    # B_BR: 1x1
    # C_TR: 4x1
    add_schoolbook(4, 1, 1, (0,4), (4,4), (0,4))
    
    # 5. A_BL B_TL
    # A_BL: 1x4
    # B_TL: 4x4
    # C_BL: 1x4
    add_schoolbook(1, 4, 4, (4,0), (0,0), (4,0))
    
    # 6. A_BR B_BL
    # A_BR: 1x1
    # B_BL: 1x4
    # C_BL: 1x4
    add_schoolbook(1, 1, 4, (4,4), (4,0), (4,0))
    
    # 7. A_BL B_TR
    # A_BL: 1x4
    # B_TR: 4x1
    # C_BR: 1x1
    add_schoolbook(1, 4, 1, (4,0), (0,4), (4,4))
    
    # 8. A_BR B_BR
    # A_BR: 1x1
    # B_BR: 1x1
    # C_BR: 1x1
    add_schoolbook(1, 1, 1, (4,4), (4,4), (4,4))
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
