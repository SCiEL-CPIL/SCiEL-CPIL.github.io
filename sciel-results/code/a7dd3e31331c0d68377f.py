import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank-7 Strassen decomposition for 2x2x2
    # U, V, W shapes (4, 7)
    U_22 = np.zeros((4, 7))
    V_22 = np.zeros((4, 7))
    W_22 = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U_22[0, 0] = 1; U_22[3, 0] = 1
    V_22[0, 0] = 1; V_22[3, 0] = 1
    W_22[0, 0] = 1 # C00
    W_22[3, 0] = 1 # C11
    
    # M2 = (c+d)e
    U_22[2, 1] = 1; U_22[3, 1] = 1
    V_22[0, 1] = 1
    W_22[1, 1] = 1 # C10
    W_22[3, 1] = -1 # C11
    
    # M3 = a(f-h)
    U_22[0, 2] = 1
    V_22[1, 2] = 1; V_22[3, 2] = -1
    W_22[2, 2] = 1 # C01
    W_22[3, 2] = 1 # C11
    
    # M4 = d(g-f)
    U_22[3, 3] = 1
    V_22[2, 3] = 1; V_22[1, 3] = -1
    W_22[0, 3] = 1 # C00
    W_22[1, 3] = 1 # C10
    
    # M5 = (a+b)h
    U_22[0, 4] = 1; U_22[1, 4] = 1
    V_22[3, 4] = 1
    W_22[0, 4] = -1 # C00
    W_22[2, 4] = 1 # C01
    
    # M6 = (c-a)(e+f)
    U_22[2, 5] = 1; U_22[0, 5] = -1
    V_22[0, 5] = 1; V_22[1, 5] = 1
    W_22[3, 5] = 1 # C11
    
    # M7 = (b-d)(g+h)
    U_22[1, 6] = 1; U_22[3, 6] = -1
    V_22[2, 6] = 1; V_22[3, 6] = 1
    W_22[0, 6] = 1 # C00
    
    # Rank-49 decomposition for 4x4x4
    # U_444, V_444, W_444 shapes (16, 49)
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # Block Strassen coefficients
    # U_block[b, r], V_block[b, r], W_block[b, r]
    # b in 0..3 (0:00, 1:01, 2:10, 3:11)
    # r in 0..6
    U_block = np.zeros((4, 7))
    V_block = np.zeros((4, 7))
    W_block = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U_block[0, 0] = 1; U_block[3, 0] = 1
    V_block[0, 0] = 1; V_block[3, 0] = 1
    W_block[0, 0] = 1 # C00
    W_block[3, 0] = 1 # C11
    
    # M2 = (A10+A11)B00
    U_block[2, 1] = 1; U_block[3, 1] = 1
    V_block[0, 1] = 1
    W_block[2, 1] = 1 # C10
    W_block[3, 1] = -1 # C11
    
    # M3 = A00(B01-B11)
    U_block[0, 2] = 1
    V_block[1, 2] = 1; V_block[3, 2] = -1
    W_block[1, 2] = 1 # C01
    W_block[3, 2] = 1 # C11
    
    # M4 = A11(B10-B00)
    U_block[3, 3] = 1
    V_block[2, 3] = 1; V_block[0, 3] = -1
    W_block[0, 3] = 1 # C00
    W_block[2, 3] = 1 # C10
    
    # M5 = (A00+A01)B11
    U_block[0, 4] = 1; U_block[1, 4] = 1
    V_block[3, 4] = 1
    W_block[0, 4] = -1 # C00
    W_block[1, 4] = 1 # C01
    
    # M6 = (A10-A00)(B00+B01)
    U_block[2, 5] = 1; U_block[0, 5] = -1
    V_block[0, 5] = 1; V_block[1, 5] = 1
    W_block[3, 5] = 1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U_block[1, 6] = 1; U_block[3, 6] = -1
    V_block[2, 6] = 1; V_block[3, 6] = 1
    W_block[0, 6] = 1 # C00
    
    # Combine
    for r in range(7):
        for s in range(7):
            R = r * 7 + s
            # U_444
            for i_block in range(4):
                if U_block[i_block, r] != 0:
                    for i_local in range(4):
                        # i_block corresponds to 2x2 block
                        # i_local is index within 2x2 block (0:0, 1:1, 2:2, 3:3)
                        # U_22 index is i_local
                        U_444[i_block * 4 + i_local, R] = U_22[i_local, s] * U_block[i_block, r]
                        
            # V_444
            for j_block in range(4):
                if V_block[j_block, r] != 0:
                    for j_local in range(4):
                        V_444[j_block * 4 + j_local, R] = V_22[j_local, s] * V_block[j_block, r]
                        
            # W_444
            for k_block in range(4):
                if W_block[k_block, r] != 0:
                    for k_local in range(4):
                        W_444[k_block * 4 + k_local, R] = W_22[k_local, s] * W_block[k_block, r]

    # Now construct for 4x4x8
    # Rank = 49 + 49 = 98
    U_448 = np.zeros((16, 98))
    V_448 = np.zeros((16, 98))
    W_448 = np.zeros((32, 98))
    
    # Left half (cols 0-3 of B, cols 0-3 of C)
    U_448[:, 0:49] = U_444
    V_448[:, 0:49] = V_444
    W_448[0:16, 0:49] = W_444
    
    # Right half (cols 4-7 of B, cols 4-7 of C)
    # B indices for right half: j*p+k. p=8.
    # j in 0..3, k in 4..7.
    # V_444 indices are j*4+k_local.
    # We need to map V_444 to V_448.
    # V_448 index is j*8+k.
    # For right half, k = 4 + k_local.
    for j in range(4):
        for k_local in range(4):
            for R in range(49):
                V_448[j * 8 + 4 + k_local, 49 + R] = V_444[j * 4 + k_local, R]
                
    # W indices for right half: k*n+i. n=4.
    # W_448 index is k*4+i.
    # For right half, k in 4..7.
    # W_444 index is k_local*4+i.
    # We need to map W_444 to W_448.
    # W_448 index is k*4+i = (4+k_local)*4+i = 16 + k_local*4+i.
    for k_local in range(4):
        for i in range(4):
            for R in range(49):
                W_448[16 + k_local * 4 + i, 49 + R] = W_444[k_local * 4 + i, R]
                
    # U is same for both halves
    U_448[:, 49:98] = U_444
    
    # Convert to lists
    U_list = U_448.tolist()
    V_list = V_448.tolist()
    W_list = W_448.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
