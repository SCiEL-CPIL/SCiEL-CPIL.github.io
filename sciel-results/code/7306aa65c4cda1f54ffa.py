import numpy as np

def construct():
    # Dimensions
    n, m, p = 3, 4, 5
    
    # We will build the decomposition by assembling smaller decompositions.
    # The total rank will be 56.
    
    # Helper to create empty factor matrices for a given rank
    def make_factors(rank, shape_u, shape_v, shape_w):
        U = np.zeros((shape_u, rank))
        V = np.zeros((shape_v, rank))
        W = np.zeros((shape_w, rank))
        return U, V, W

    # 1. M(2, 2, 2) - Strassen Rank 7
    # A (2x2), B (2x2), C (2x2)
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    def get_strassen():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # m1 = (A00 + A11)(B00 + B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # m2 = (A10 + A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # m3 = A00(B10 - B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # m4 = A11(B01 - B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1
        
        # m5 = (A00 + A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        
        # m6 = (A10 - A00)(B00 + B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # m7 = (A01 - A11)(B10 + B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1
        
        # Wait, the W indices for Strassen need to map to C00, C01, C10, C11
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # My W assignments above were just 1s. I need to set the coefficients in W.
        # Let's re-initialize W and set the linear combinations.
        W = np.zeros((4, 7))
        
        # C00 (index 0)
        W[0, 0] = 1  # m1
        W[0, 3] = 1  # m4
        W[0, 4] = -1 # -m5
        W[0, 6] = 1  # m7
        
        # C01 (index 1)
        W[1, 2] = 1  # m3
        W[1, 4] = 1  # m5
        
        # C10 (index 2)
        W[2, 1] = 1  # m2
        W[2, 3] = 1  # m4
        
        # C11 (index 3)
        W[3, 0] = 1  # m1
        W[3, 1] = -1 # -m2
        W[3, 2] = 1  # m3
        W[3, 5] = 1  # m6
        
        return U, V, W

    # 2. M(1, 2, 2) - Schoolbook Rank 4
    # A (1x2), B (2x2), C (1x2)
    # U shape (2, 4), V shape (4, 4), W shape (2, 4)
    def get_122():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        
        # m1 = A0 B00
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # m2 = A1 B10
        U[1, 1] = 1
        V[2, 1] = 1
        W[0, 1] = 1
        
        # m3 = A0 B01
        U[0, 2] = 1
        V[1, 2] = 1
        W[1, 2] = 1
        
        # m4 = A1 B11
        U[1, 3] = 1
        V[3, 3] = 1
        W[1, 3] = 1
        
        return U, V, W

    # 3. M(3, 4, 1) - Schoolbook Rank 12
    # A (3x4), B (4x1), C (3x1)
    # U shape (12, 12), V shape (4, 12), W shape (3, 12)
    def get_341():
        U = np.zeros((12, 12))
        V = np.zeros((4, 12))
        W = np.zeros((3, 12))
        
        col = 0
        for i in range(3):
            for j in range(4):
                a_idx = i * 4 + j
                b_idx = j
                c_idx = i
                
                U[a_idx, col] = 1
                V[b_idx, col] = 1
                W[c_idx, col] = 1
                col += 1
                
        return U, V, W

    # 4. Assemble M(3, 2, 2) from M(2, 2, 2) and M(1, 2, 2)
    # A (3x2) split into A_top (2x2) and A_bot (1x2)
    # B (2x2) is same
    # C (3x2) split into C_top (2x2) and C_bot (1x2)
    
    def get_322():
        U_s, V_s, W_s = get_strassen()
        U_1, V_1, W_1 = get_122()
        
        # Total rank 7 + 4 = 11
        U = np.zeros((6, 11)) # 3*2
        V = np.zeros((4, 11)) # 2*2
        W = np.zeros((6, 11)) # 3*2
        
        # Copy Strassen (top 2 rows of A, all 2 rows of B, top 2 rows of C)
        # A indices: 0,1 (top), 2,3 (bot) -- Wait, A is 3x2.
        # A_top is rows 0,1. A_bot is row 2.
        # B is 2x2.
        # C_top is rows 0,1. C_bot is row 2.
        
        # Strassen uses A[0..3] (2x2), B[0..3] (2x2), C[0..3] (2x2)
        # Map Strassen A to U rows 0,1,4,5? No.
        # A_top is 2x2. Indices in 3x2 A:
        # A[0,0]=0, A[0,1]=1
        # A[1,0]=2, A[1,1]=3
        # A[2,0]=4, A[2,1]=5
        # Strassen A indices 0,1,2,3 map to 0,1,2,3.
        # So U[0:4, 0:7] = U_s.
        U[0:4, 0:7] = U_s
        
        # B is 2x2. Indices 0,1,2,3.
        V[0:4, 0:7] = V_s
        
        # C_top is 2x2. Indices 0,1,2,3 in C (3x2).
        # C[0,0]=0, C[0,1]=1
        # C[1,0]=2, C[1,1]=3
        W[0:4, 0:7] = W_s
        
        # Copy 122 (bot 1 row of A, all 2 rows of B, bot 1 row of C)
        # 122 A is 1x2. Indices 0,1.
        # Map to U rows 4,5.
        U[4:6, 7:11] = U_1
        
        # 122 B is 2x2. Indices 0,1,2,3.
        V[0:4, 7:11] = V_1
        
        # 122 C is 1x2. Indices 0,1.
        # Map to W rows 4,5.
        W[4:6, 7:11] = W_1
        
        return U, V, W

    # 5. Assemble M(3, 4, 2) from two M(3, 2, 2)
    # A (3x4) split into A1 (3x2) and A2 (3x2)
    # B (4x2) split into B1 (2x2) and B2 (2x2)
    # C (3x2)
    
    def get_342():
        U_1, V_1, W_1 = get_322()
        U_2, V_2, W_2 = get_322()
        
        # Total rank 11 + 11 = 22
        U = np.zeros((12, 22)) # 3*4
        V = np.zeros((8, 22)) # 4*2
        W = np.zeros((6, 22)) # 3*2
        
        # A1 is cols 0,1. A2 is cols 2,3.
        # A1 indices in 3x4:
        # 0,1 (row 0)
        # 4,5 (row 1)
        # 8,9 (row 2)
        # U_1 indices 0..5 map to these.
        
        # A2 indices in 3x4:
        # 2,3 (row 0)
        # 6,7 (row 1)
        # 10,11 (row 2)
        # U_2 indices 0..5 map to these.
        
        a1_indices = [0, 1, 4, 5, 8, 9]
        a2_indices = [2, 3, 6, 7, 10, 11]
        
        U[a1_indices, 0:11] = U_1
        U[a2_indices, 11:22] = U_2
        
        # B1 is rows 0,1 of B (4x2).
        # B indices:
        # 0,1 (row 0)
        # 2,3 (row 1)
        # V_1 indices 0..3 map to 0,1,2,3.
        
        # B2 is rows 2,3 of B (4x2).
        # 4,5 (row 2)
        # 6,7 (row 3)
        # V_2 indices 0..3 map to 4,5,6,7.
        
        b1_indices = [0, 1, 2, 3]
        b2_indices = [4, 5, 6, 7]
        
        V[b1_indices, 0:11] = V_1
        V[b2_indices, 11:22] = V_2
        
        # C is same for both.
        W[:, 0:11] = W_1
        W[:, 11:22] = W_2
        
        return U, V, W

    # 6. Assemble M(3, 4, 5)
    # B (4x5) split into B1 (4x2), B2 (4x2), B3 (4x1)
    # C (3x5) split into C1 (3x2), C2 (3x2), C3 (3x1)
    
    U_1, V_1, W_1 = get_342()
    U_2, V_2, W_2 = get_342()
    U_3, V_3, W_3 = get_341()
    
    # Total rank 22 + 22 + 12 = 56
    U = np.zeros((12, 56))
    V = np.zeros((20, 56))
    W = np.zeros((15, 56))
    
    # A is same for all.
    U[:, 0:22] = U_1
    U[:, 22:44] = U_2
    U[:, 44:56] = U_3
    
    # B1 is cols 0,1. B2 is cols 2,3. B3 is col 4.
    # B indices in 4x5:
    # B1:
    # 0,1 (row 0)
    # 5,6 (row 1)
    # 10,11 (row 2)
    # 15,16 (row 3)
    b1_indices = [0, 1, 5, 6, 10, 11, 15, 16]
    
    # B2:
    # 2,3 (row 0)
    # 7,8 (row 1)
    # 12,13 (row 2)
    # 17,18 (row 3)
    b2_indices = [2, 3, 7, 8, 12, 13, 17, 18]
    
    # B3:
    # 4 (row 0)
    # 9 (row 1)
    # 14 (row 2)
    # 19 (row 3)
    b3_indices = [4, 9, 14, 19]
    
    V[b1_indices, 0:22] = V_1
    V[b2_indices, 22:44] = V_2
    V[b3_indices, 44:56] = V_3
    
    # C1 is cols 0,1. C2 is cols 2,3. C3 is col 4.
    # C indices in 3x5 (p*n = 5*3 = 15, index k*n + i)
    # C1 (k=0,1):
    # k=0: i=0,1,2 -> 0,1,2
    # k=1: i=0,1,2 -> 3,4,5
    c1_indices = [0, 1, 2, 3, 4, 5]
    
    # C2 (k=2,3):
    # k=2: i=0,1,2 -> 6,7,8
    # k=3: i=0,1,2 -> 9,10,11
    c2_indices = [6, 7, 8, 9, 10, 11]
    
    # C3 (k=4):
    # k=4: i=0,1,2 -> 12,13,14
    c3_indices = [12, 13, 14]
    
    W[c1_indices, 0:22] = W_1
    W[c2_indices, 22:44] = W_2
    W[c3_indices, 44:56] = W_3
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
