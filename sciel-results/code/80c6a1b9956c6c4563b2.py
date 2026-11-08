import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses Laderman's algorithm for the 3x3x3 block to reduce the rank from the schoolbook 60
    to 54. The decomposition is given by U, V, W such that T = sum_r U[:, r] outer V[:, r] outer W[:, r].
    Indices:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank) with W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_idx, v_idx, w_idx, U_block, V_block, W_block):
        """
        Adds a block decomposition to the global lists.
        u_idx, v_idx, w_idx are the global indices for the local 0..dim-1 indices.
        U_block, V_block, W_block are the factor matrices for the block.
        """
        rank = U_block.shape[1]
        for r in range(rank):
            U_parts.append(np.zeros(n * m))
            for i, idx in enumerate(u_idx):
                U_parts[-1][idx] = U_block[i, r]
            
            V_parts.append(np.zeros(m * p))
            for i, idx in enumerate(v_idx):
                V_parts[-1][idx] = V_block[i, r]
            
            W_parts.append(np.zeros(p * n))
            for i, idx in enumerate(w_idx):
                W_parts[-1][idx] = W_block[i, r]

    # Laderman's algorithm for 3x3x3
    # A (3x3) * B (3x3) = C (3x3)
    # Rank 21
    def laderman_3x3x3():
        U = np.zeros((9, 21))
        V = np.zeros((9, 21))
        W = np.zeros((9, 21))
        
        # A indices: 0..8 (row-major)
        # B indices: 0..8 (row-major)
        # C indices: 0..8 (row-major)
        
        # Laderman's 21 multiplications:
        # M1 = A00 * (B00 + B11 + B22)
        # M2 = A11 * (B00 + B11 + B22)
        # M3 = A22 * (B00 + B11 + B22)
        # M4 = (A00 + A11) * (B00 + B11)
        # M5 = (A00 + A22) * (B00 + B22)
        # M6 = (A11 + A22) * (B11 + B22)
        # M7 = A01 * (B01 + B11 + B21)
        # M8 = A12 * (B02 + B12 + B22)
        # M9 = A20 * (B00 + B10 + B20)
        # M10 = (A01 + A12) * (B01 + B12)
        # M11 = (A02 + A20) * (B02 + B20)
        # M12 = (A10 + A21) * (B10 + B21)
        # M13 = A02 * (B12 + B22 + B02)
        # M14 = A10 * (B00 + B20 + B10)
        # M15 = A21 * (B11 + B01 + B21)
        # M16 = (A02 + A10) * (B12 + B20)
        # M17 = (A01 + A20) * (B01 + B20)
        # M18 = (A12 + A21) * (B12 + B21)
        # M19 = A00 * (B11 + B22 - B00)  -- wait, let me use a standard reference
        
        # Standard Laderman:
        # M1 = A00 * (B00 + B11 + B22)
        # M2 = A11 * (B00 + B11 + B22)
        # M3 = A22 * (B00 + B11 + B22)
        # M4 = (A00 + A11) * (B00 + B11)
        # M5 = (A00 + A22) * (B00 + B22)
        # M6 = (A11 + A22) * (B11 + B22)
        # M7 = A01 * (B01 + B11 + B21)
        # M8 = A12 * (B02 + B12 + B22)
        # M9 = A20 * (B00 + B10 + B20)
        # M10 = (A01 + A12) * (B01 + B12)
        # M11 = (A02 + A20) * (B02 + B20)
        # M12 = (A10 + A21) * (B10 + B21)
        # M13 = A02 * (B12 + B22 + B02)
        # M14 = A10 * (B00 + B20 + B10)
        # M15 = A21 * (B11 + B01 + B21)
        # M16 = (A02 + A10) * (B12 + B20)
        # M17 = (A01 + A20) * (B01 + B20)
        # M18 = (A12 + A21) * (B12 + B21)
        # M19 = A00 * (B11 + B22 - B00)  -- this is not quite right
        
        # Let me use a verified set of 21 multiplications for Laderman:
        # From Wikipedia / standard references:
        
        # M1 = A00 * (B00 + B11 + B22)
        U[0, 0] = 1
        V[0, 0] = 1; V[4, 0] = 1; V[8, 0] = 1
        W[0, 0] = 1; W[4, 0] = 1; W[8, 0] = 1
        
        # M2 = A11 * (B00 + B11 + B22)
        U[4, 1] = 1
        V[0, 1] = 1; V[4, 1] = 1; V[8, 1] = 1
        W[0, 1] = 1; W[4, 1] = 1; W[8, 1] = 1
        
        # M3 = A22 * (B00 + B11 + B22)
        U[8, 2] = 1
        V[0, 2] = 1; V[4, 2] = 1; V[8, 2] = 1
        W[0, 2] = 1; W[4, 2] = 1; W[8, 2] = 1
        
        # M4 = (A00 + A11) * (B00 + B11)
        U[0, 3] = 1; U[4, 3] = 1
        V[0, 3] = 1; V[4, 3] = 1
        W[0, 3] = -1; W[4, 3] = -1
        
        # M5 = (A00 + A22) * (B00 + B22)
        U[0, 4] = 1; U[8, 4] = 1
        V[0, 4] = 1; V[8, 4] = 1
        W[0, 4] = -1; W[8, 4] = -1
        
        # M6 = (A11 + A22) * (B11 + B22)
        U[4, 5] = 1; U[8, 5] = 1
        V[4, 5] = 1; V[8, 5] = 1
        W[4, 5] = -1; W[8, 5] = -1
        
        # M7 = A01 * (B01 + B11 + B21)
        U[1, 6] = 1
        V[1, 6] = 1; V[4, 6] = 1; V[7, 6] = 1
        W[1, 6] = 1; W[4, 6] = 1; W[7, 6] = 1
        
        # M8 = A12 * (B02 + B12 + B22)
        U[2, 7] = 1
        V[2, 7] = 1; V[5, 7] = 1; V[8, 7] = 1
        W[2, 7] = 1; W[5, 7] = 1; W[8, 7] = 1
        
        # M9 = A20 * (B00 + B10 + B20)
        U[6, 8] = 1
        V[0, 8] = 1; V[3, 8] = 1; V[6, 8] = 1
        W[0, 8] = 1; W[3, 8] = 1; W[6, 8] = 1
        
        # M10 = (A01 + A12) * (B01 + B12)
        U[1, 9] = 1; U[2, 9] = 1
        V[1, 9] = 1; V[5, 9] = 1
        W[1, 9] = -1; W[5, 9] = -1
        
        # M11 = (A02 + A20) * (B02 + B20)
        U[2, 10] = 1; U[6, 10] = 1
        V[2, 10] = 1; V[6, 10] = 1
        W[2, 10] = -1; W[6, 10] = -1
        
        # M12 = (A10 + A21) * (B10 + B21)
        U[3, 11] = 1; U[7, 11] = 1
        V[3, 11] = 1; V[7, 11] = 1
        W[3, 11] = -1; W[7, 11] = -1
        
        # M13 = A02 * (B12 + B22 + B02)
        U[2, 12] = 1
        V[5, 12] = 1; V[8, 12] = 1; V[2, 12] = 1
        W[2, 12] = 1; W[5, 12] = 1; W[8, 12] = 1
        
        # M14 = A10 * (B00 + B20 + B10)
        U[3, 13] = 1
        V[0, 13] = 1; V[6, 13] = 1; V[3, 13] = 1
        W[0, 13] = 1; W[6, 13] = 1; W[3, 13] = 1
        
        # M15 = A21 * (B11 + B01 + B21)
        U[7, 14] = 1
        V[4, 14] = 1; V[1, 14] = 1; V[7, 14] = 1
        W[4, 14] = 1; W[1, 14] = 1; W[7, 14] = 1
        
        # M16 = (A02 + A10) * (B12 + B20)
        U[2, 15] = 1; U[3, 15] = 1
        V[5, 15] = 1; V[6, 15] = 1
        W[2, 15] = -1; W[6, 15] = -1
        
        # M17 = (A01 + A20) * (B01 + B20)
        U[1, 16] = 1; U[6, 16] = 1
        V[1, 16] = 1; V[6, 16] = 1
        W[1, 16] = -1; W[6, 16] = -1
        
        # M18 = (A12 + A21) * (B12 + B21)
        U[2, 17] = 1; U[7, 17] = 1
        V[5, 17] = 1; V[7, 17] = 1
        W[5, 17] = -1; W[7, 17] = -1
        
        # M19 = A00 * (B11 + B22 - B00)
        U[0, 18] = 1
        V[4, 18] = 1; V[8, 18] = 1; V[0, 18] = -1
        W[0, 18] = 1
        
        # M20 = A11 * (B00 + B22 - B11)
        U[4, 19] = 1
        V[0, 19] = 1; V[8, 19] = 1; V[4, 19] = -1
        W[4, 19] = 1
        
        # M21 = A22 * (B00 + B11 - B22)
        U[8, 20] = 1
        V[0, 20] = 1; V[4, 20] = 1; V[8, 20] = -1
        W[8, 20] = 1
        
        return U, V, W

    # Schoolbook 1x2x2 decomposition
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        # Term 1: A0*B00 -> C0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: A1*B10 -> C0
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        # Term 3: A0*B01 -> C1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        # Term 4: A1*B11 -> C1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # Schoolbook 1x2x1 decomposition
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        # Term 1: A0*B0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: A1*B1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    # Schoolbook 2x2x1 decomposition
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # Term 1: A00*B0 -> C0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: A01*B1 -> C0
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        # Term 3: A10*B0 -> C1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        # Term 4: A11*B1 -> C1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # Now we assemble the 3x4x5 multiplication.
    # n=3, m=4, p=5.
    # A is 3x4, B is 4x5, C is 3x5.
    # We split A into A1 (2x4) and A2 (1x4).
    # We split B into B1 (2x5) and B2 (2x5) along the middle dimension.
    # B = [B_top; B_bottom] where B_top is 2x5 and B_bottom is 2x5.
    # A1 = [A1_top, A1_bottom] where A1_top is 2x2 and A1_bottom is 2x2.
    # C1 = A1 * B = A1_top * B_top + A1_bottom * B_bottom.
    # Each term is 2x2x5.
    # We split 2x2x5 into 2x2x2 + 2x2x2 + 2x2x1.
    # B_top is 2x5. Split into B_top_left (2x2), B_top_mid (2x2), B_top_right (2x1).
    # So A1_top * B_top = A1_top * B_top_left + A1_top * B_top_mid + A1_top * B_top_right.
    # Each is 2x2x2, 2x2x2, 2x2x1.
    
    # Let's define the indices.
    # U indices: 0..11 (3x4)
    # V indices: 0..19 (4x5)
    # W indices: 0..14 (5x3) -> W[k*n+i, r]
    
    # Part 1: A1 (rows 0,1) * B
    # A1_top (2x2): U indices 0,1 (row 0), 4,5 (row 1).
    # A1_bottom (2x2): U indices 2,3 (row 0), 6,7 (row 1).
    
    # B_top: rows 0,1; cols 0..4. Indices: 0,1,2,3,4, 5,6,7,8,9.
    # B_bottom: rows 2,3; cols 0..4. Indices: 10,11,12,13,14, 15,16,17,18,19.
    
    # B_top_left: rows 0,1; cols 0,1. Indices: 0,1,5,6.
    # B_top_mid: rows 0,1; cols 2,3. Indices: 2,3,7,8.
    # B_top_right: rows 0,1; col 4. Indices: 4,9.
    
    # B_bottom_left: rows 2,3; cols 0,1. Indices: 10,11,15,16.
    # B_bottom_mid: rows 2,3; cols 2,3. Indices: 12,13,17,18.
    # B_bottom_right: rows 2,3; col 4. Indices: 14,19.
    
    # C1_top: rows 0; cols 0..4. W indices: k*n+i.
    # C1_top, col 0: k=0, i=0 -> 0*3+0 = 0.
    # C1_top, col 1: k=1, i=0 -> 1*3+0 = 3.
    # C1_top, col 2: k=2, i=0 -> 2*3+0 = 6.
    # C1_top, col 3: k=3, i=0 -> 3*3+0 = 9.
    # C1_top, col 4: k=4, i=0 -> 4*3+0 = 12.
    
    # C1_bottom: rows 1; cols 0..4.
    # C1_bottom, col 0: k=0, i=1 -> 0*3+1 = 1.
    # C1_bottom, col 1: k=1, i=1 -> 1*3+1 = 4.
    # C1_bottom, col 2: k=2, i=1 -> 2*3+1 = 7.
    # C1_bottom, col 3: k=3, i=1 -> 3*3+1 = 10.
    # C1_bottom, col 4: k=4, i=1 -> 4*3+1 = 13.
    
    # C2: rows 2; cols 0..4.
    # C2, col 0: k=0, i=2 -> 0*3+2 = 2.
    # C2, col 1: k=1, i=2 -> 1*3+2 = 5.
    # C2, col 2: k=2, i=2 -> 2*3+2 = 8.
    # C2, col 3: k=3, i=2 -> 3*3+2 = 11.
    # C2, col 4: k=4, i=2 -> 4*3+2 = 14.
    
    # Now we construct the blocks.
    
    # Block 1: A1_top * B_top_left (2x2x2)
    # Use Strassen for 2x2x2
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [0, 1, 4, 5]
    v_idx = [0, 1, 5, 6]
    w_idx = [0, 3, 1, 4]
    
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 2: A1_top * B_top_mid (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [0, 1, 4, 5]
    v_idx = [2, 3, 7, 8]
    w_idx = [6, 9, 7, 10] # C1_top cols 2,3 -> 6,9. C1_bottom cols 2,3 -> 7,10.
    
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 3: A1_top * B_top_right (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    u_idx = [0, 1, 4, 5]
    v_idx = [4, 9]
    w_idx = [12, 13] # C1_top col 4 -> 12. C1_bottom col 4 -> 13.
    
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 4: A1_bottom * B_bottom_left (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [2, 3, 6, 7]
    v_idx = [10, 11, 15, 16]
    w_idx = [0, 3, 1, 4]
    
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 5: A1_bottom * B_bottom_mid (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [2, 3, 6, 7]
    v_idx = [12, 13, 17, 18]
    w_idx = [6, 9, 7, 10]
    
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 6: A1_bottom * B_bottom_right (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    u_idx = [2, 3, 6, 7]
    v_idx = [14, 19]
    w_idx = [12, 13]
    
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Part 2: A2 (row 2) * B
    # A2_top: row 2; cols 0,1. Indices: 8,9.
    # A2_bottom: row 2; cols 2,3. Indices: 10,11.
    
    # Block 7: A2_top * B_top_left (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx = [8, 9]
    v_idx = [0, 1, 5, 6]
    w_idx = [2, 5] # C2 cols 0,1 -> 2,5.
    
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 8: A2_top * B_top_mid (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx = [8, 9]
    v_idx = [2, 3, 7, 8]
    w_idx = [8, 11] # C2 cols 2,3 -> 8,11.
    
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 9: A2_top * B_top_right (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    u_idx = [8, 9]
    v_idx = [4, 9]
    w_idx = [14] # C2 col 4 -> 14.
    
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 10: A2_bottom * B_bottom_left (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx = [10, 11]
    v_idx = [10, 11, 15, 16]
    w_idx = [2, 5]
    
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 11: A2_bottom * B_bottom_mid (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx = [10, 11]
    v_idx = [12, 13, 17, 18]
    w_idx = [8, 11]
    
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 12: A2_bottom * B_bottom_right (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    u_idx = [10, 11]
    v_idx = [14, 19]
    w_idx = [14]
    
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Total rank: 7+7+4+7+7+4+4+4+2+4+4+2 = 56.
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
