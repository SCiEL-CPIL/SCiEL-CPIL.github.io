import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # 1. Generate 2x2x2 Strassen (Rank 7)
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # m1
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    # m2
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = 1
    # m3
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    # m4
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[3, 3] = 1
    # m5
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = 1; W_2[1, 4] = -1
    # m6
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[2, 5] = 1
    # m7
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1; W_2[1, 6] = 1

    # 2. Generate 4x4x4 Strassen (Rank 49)
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    # Blocks of A, B, C
    # A = [A00 A01; A10 A11]
    # B = [B00 B01; B10 B11]
    # C = [C00 C01; C10 C11]
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00 -> C10, C11
    # M3 = A00(B10-B00) -> C01, C11
    # M4 = A11(B01-B10) -> C00, C11
    # M5 = (A00+A01)B11 -> C00, C01
    # M6 = (A10-A00)(B00+B01) -> C10
    # M7 = (A01-A11)(B10+B11) -> C00, C01
    
    def add_strassen_block(U_t, V_t, W_t, a_blocks, b_blocks, c_blocks, offset):
        for s in range(7):
            col = offset + s
            for (br, bc, sign) in a_blocks:
                for i in range(2):
                    for j in range(2):
                        u_idx = (2*br+i)*4 + (2*bc+j)
                        u_loc = i*2 + j
                        U_t[u_idx, col] += sign * U_2[u_loc, s]
            for (br, bc, sign) in b_blocks:
                for i in range(2):
                    for j in range(2):
                        v_idx = (2*br+i)*4 + (2*bc+j)
                        v_loc = i*2 + j
                        V_t[v_idx, col] += sign * V_2[v_loc, s]
            for (br, bc, sign) in c_blocks:
                for i in range(2):
                    for j in range(2):
                        w_idx = (2*br+i)*4 + (2*bc+j)
                        w_loc = i*2 + j
                        W_t[w_idx, col] += sign * W_2[w_loc, s]

    add_strassen_block(U_4, V_4, W_4,
                       [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], 0)
    add_strassen_block(U_4, V_4, W_4,
                       [(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,1)], 7)
    add_strassen_block(U_4, V_4, W_4,
                       [(0,0,1)], [(1,0,1), (0,0,-1)], [(0,1,1), (1,1,1)], 14)
    add_strassen_block(U_4, V_4, W_4,
                       [(1,1,1)], [(0,1,1), (1,0,-1)], [(0,0,1), (1,1,1)], 21)
    add_strassen_block(U_4, V_4, W_4,
                       [(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,1), (0,1,-1)], 28)
    add_strassen_block(U_4, V_4, W_4,
                       [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,0,1)], 35)
    add_strassen_block(U_4, V_4, W_4,
                       [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (0,1,1)], 42)

    # 3. Generate 4x4x1 Schoolbook (Rank 16)
    U_1 = np.zeros((16, 16))
    V_1 = np.zeros((4, 16)) # B is 4x1
    W_1 = np.zeros((4, 16)) # C is 1x4 -> 4x1 in W? No, W is (p*n, rank). p=1, n=4. So 4x16.
    
    for i in range(4):
        for j in range(4):
            r = i*4 + j
            U_1[i*4 + j, r] = 1
            V_1[j, r] = 1
            W_1[i, r] = 1 # W index is k*n + i. k=0. So i.

    # 4. Assemble 4x4x9
    total_rank = 49 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Strassen part (cols 0-48)
    U[:, 0:49] = U_4
    
    for r in range(49):
        for j in range(4):
            for k in range(8):
                if k < 4:
                    V[j*9 + k, r] = V_4[j*4 + k, r]
                else:
                    V[j*9 + k, r] = V_4[j*4 + (k-4), r]
            # k=8 is 0
            
        for i in range(4):
            for k in range(8):
                if k < 4:
                    W[k*4 + i, r] = W_4[k*4 + i, r]
                else:
                    W[k*4 + i, r] = W_4[(k-4)*4 + i, r]
            # k=8 is 0

    # Schoolbook part (cols 49-64)
    U[:, 49:65] = U_1
    
    for r in range(16):
        for j in range(4):
            V[j*9 + 8, 49+r] = V_1[j, r]
            
        for i in range(4):
            W[8*4 + i, 49+r] = W_1[i, r]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
