import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses a block decomposition approach:
    - Split A (3x4) into A1 (2x4) and A2 (1x4)
    - Split B (4x5) into B1 (2x5) and B2 (2x5)
    - Each 2x2x5 block is decomposed using Strassen (2x2x2) + schoolbook (2x2x1)
    - Each 1x2x5 block is decomposed using schoolbook (1x2x2) + schoolbook (1x2x1)
    
    Total rank: 56
    """
    n, m, p = DIMENSIONS
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Helper: Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    # Helper: Schoolbook 2x2x1 decomposition (rank 4)
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    # Helper: Schoolbook 1x2x2 decomposition (rank 4)
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    # Helper: Schoolbook 1x2x1 decomposition (rank 2)
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W
    
    # Add a block to the global decomposition
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            U_parts.append(np.zeros(n * m))
            for i, idx in enumerate(u_idx):
                U_parts[-1][idx] = U_blk[i, r]
            
            V_parts.append(np.zeros(m * p))
            for i, idx in enumerate(v_idx):
                V_parts[-1][idx] = V_blk[i, r]
            
            W_parts.append(np.zeros(p * n))
            for i, idx in enumerate(w_idx):
                W_parts[-1][idx] = W_blk[i, r]
    
    # Part 1: A1 (rows 0,1) * B
    # A1_top: rows 0,1; cols 0,1. U indices: 0,1,4,5
    # A1_bottom: rows 0,1; cols 2,3. U indices: 2,3,6,7
    
    # B_top: rows 0,1; cols 0..4. V indices: 0,1,2,3,4,5,6,7,8,9
    # B_bottom: rows 2,3; cols 0..4. V indices: 10,11,12,13,14,15,16,17,18,19
    
    # W indices: C[i,k] -> W[k*n+i]
    # C row 0: k=0..4 -> 0,3,6,9,12
    # C row 1: k=0..4 -> 1,4,7,10,13
    # C row 2: k=0..4 -> 2,5,8,11,14
    
    # Block 1: A1_top * B_top_left (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([0, 1, 4, 5], [0, 1, 5, 6], [0, 3, 1, 4], U_blk, V_blk, W_blk)
    
    # Block 2: A1_top * B_top_mid (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([0, 1, 4, 5], [2, 3, 7, 8], [6, 9, 7, 10], U_blk, V_blk, W_blk)
    
    # Block 3: A1_top * B_top_right (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block([0, 1, 4, 5], [4, 9], [12, 13], U_blk, V_blk, W_blk)
    
    # Block 4: A1_bottom * B_bottom_left (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([2, 3, 6, 7], [10, 11, 15, 16], [0, 3, 1, 4], U_blk, V_blk, W_blk)
    
    # Block 5: A1_bottom * B_bottom_mid (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([2, 3, 6, 7], [12, 13, 17, 18], [6, 9, 7, 10], U_blk, V_blk, W_blk)
    
    # Block 6: A1_bottom * B_bottom_right (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block([2, 3, 6, 7], [14, 19], [12, 13], U_blk, V_blk, W_blk)
    
    # Part 2: A2 (row 2) * B
    # A2_top: row 2; cols 0,1. U indices: 8,9
    # A2_bottom: row 2; cols 2,3. U indices: 10,11
    
    # Block 7: A2_top * B_top_left (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block([8, 9], [0, 1, 5, 6], [2, 5], U_blk, V_blk, W_blk)
    
    # Block 8: A2_top * B_top_mid (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block([8, 9], [2, 3, 7, 8], [8, 11], U_blk, V_blk, W_blk)
    
    # Block 9: A2_top * B_top_right (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block([8, 9], [4, 9], [14], U_blk, V_blk, W_blk)
    
    # Block 10: A2_bottom * B_bottom_left (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block([10, 11], [10, 11, 15, 16], [2, 5], U_blk, V_blk, W_blk)
    
    # Block 11: A2_bottom * B_bottom_mid (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block([10, 11], [12, 13, 17, 18], [8, 11], U_blk, V_blk, W_blk)
    
    # Block 12: A2_bottom * B_bottom_right (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block([10, 11], [14, 19], [14], U_blk, V_blk, W_blk)
    
    # Total rank: 7+7+4+7+7+4+4+4+2+4+4+2 = 56
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
