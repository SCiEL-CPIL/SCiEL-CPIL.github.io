import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses Strassen's algorithm for 2x2x2 blocks and optimized 1x2x2 decompositions
    to reduce the rank.
    The decomposition is given by U, V, W such that T = sum_r U[:, r] outer V[:, r] outer W[:, r].
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
    
    # 1. Strassen 2x2x2 decomposition
    # U: 4x7, V: 4x7, W: 4x7
    # U indices: 0:A00, 1:A01, 2:A10, 3:A11
    # V indices: 0:B00, 1:B01, 2:B10, 3:B11
    # W indices: 0:C00, 1:C01, 2:C10, 3:C11
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
    
    # 2. Schoolbook 2x2x1 decomposition
    # A (2x2) * B (2x1) = C (2x1)
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # Rank 4.
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
    
    # 3. Optimized 1x2x2 decomposition
    # A (1x2) * B (2x2) = C (1x2)
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    # We can use 3 multiplications:
    # M1 = (A0+A1)B00
    # M2 = A0(B01-B11)
    # M3 = A1(B10+B11)
    # Let's check:
    # C0 = A0*B00 + A1*B10
    #     = (M1 - A1*B00) + A1*B10  -- this doesn't seem to simplify directly to M1, M2, M3 without extra terms.
    
    # Let's try another combination.
    # M1 = A0(B00+B01)
    # M2 = A1(B10+B11)
    # M3 = (A0+A1)(B00+B10) ? No.
    
    # Actually, the rank of 1x2x2 is 4. It is equivalent to two independent dot products of length 2?
    # No, C0 and C1 share A0 and A1.
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    # This is a bilinear map.
    # Tensor T = e0⊗e00⊗e0 + e1⊗e10⊗e0 + e0⊗e01⊗e1 + e1⊗e11⊗e1
    # This is a sum of 4 rank-1 tensors.
    # Can we do it in 3?
    # If we could, the tensor rank would be 3.
    # The border rank of 1x2x2 is 3?
    # Let's check the catalecticant.
    # For 1x2x2, the tensor is:
    # [B00 B01]
    # [B10 B11] multiplied by [A0, A1]^T
    # This is just a linear map from R^2 to R^2 given by the matrix B, scaled by A?
    # No, A is a vector.
    # It's A^T * B.
    # The tensor is T_{i,j,k} = delta_{i,0} delta_{j,k} ... no.
    # T_{i, j, k} where i is A index, j is B row, k is B col?
    # A is (1,2). B is (2,2). C is (1,2).
    # C_k = sum_j A_j B_{j,k}
    # T_{j,k,l} = delta_{j, l} ... no.
    # T_{j, k, l} where j is A index, k is B index, l is C index.
    # C_l = sum_j A_j B_{j,l}
    # T_{j, k, l} = delta_{j,k} delta_{k,l}? No.
    # T_{j, k, l} = 1 if k=l and j=k? No.
    # The term A_j B_{j,l} contributes to C_l.
    # So T_{j, k, l} = 1 if k=j and l=k?
    # Yes, T_{j, j, j} = 1? No.
    # A_j is index j. B_{j,l} is row j, col l.
    # So the term is A_j * B_{j,l}.
    # The tensor has non-zero entries at (j, j, l) for all j,l?
    # No, the input is A (size 2) and B (size 4).
    # A has indices 0,1.
    # B has indices 0,1 (row) and 0,1 (col).
    # C has indices 0,1.
    # C_l = A_0 B_{0,l} + A_1 B_{1,l}
    # So the tensor T has T_{0, 0, l} = 1 and T_{1, 1, l} = 1?
    # No, the B index is a single index from 0 to 3.
    # Let B_{0,0}=b0, B_{0,1}=b1, B_{1,0}=b2, B_{1,1}=b3.
    # C_0 = A_0 b0 + A_1 b2
    # C_1 = A_0 b1 + A_1 b3
    # T_{0, 0, 0} = 1
    # T_{1, 2, 0} = 1
    # T_{0, 1, 1} = 1
    # T_{1, 3, 1} = 1
    # This is a sum of 4 rank-1 tensors:
    # e0⊗e0⊗e0 + e1⊗e2⊗e0 + e0⊗e1⊗e1 + e1⊗e3⊗e1
    # Are these independent?
    # Yes.
    # So rank is at least 4.
    # And we have a decomposition of rank 4.
    # So rank is exactly 4.
    
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
    
    # 4. Schoolbook 1x2x1 decomposition
    # A (1x2) * B (2x1) = C (1x1)
    # C0 = A0*B0 + A1*B1
    # Rank 2.
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        # Term 1: A0*B0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: A1*B1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    # Now we assemble the 3x4x5 multiplication.
    # n=3, m=4, p=5.
    # A is 3x4, B is 4x5, C is 3x5.
    # We split A into A1 (2x4) and A2 (1x4).
    # We split B into B1 (2x5) and B2 (2x5) along the middle dimension?
    # No, B is 4x5. We split the rows of B into two groups of 2.
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
    # A1_top (2x2): U indices 0,1 (row 0), 4,5 (row 1). Wait, U is flattened row-major.
    # U[i*m + j].
    # Row 0: j=0,1,2,3 -> indices 0,1,2,3.
    # Row 1: j=0,1,2,3 -> indices 4,5,6,7.
    # Row 2: j=0,1,2,3 -> indices 8,9,10,11.
    
    # A1_top: rows 0,1; cols 0,1. Indices: 0,1,4,5.
    # A1_bottom: rows 0,1; cols 2,3. Indices: 2,3,6,7.
    
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
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [0, 1, 4, 5]
    v_idx = [0, 1, 5, 6]
    w_idx = [0, 3, 1, 4]
    
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 2: A1_top * B_top_mid (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [0, 1, 4, 5]
    v_idx = [2, 3, 7, 8]
    w_idx = [6, 9, 7, 10]
    
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 3: A1_top * B_top_right (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    u_idx = [0, 1, 4, 5]
    v_idx = [4, 9]
    w_idx = [12, 13]
    
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
    w_idx = [2, 5]
    
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 8: A2_top * B_top_mid (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx = [8, 9]
    v_idx = [2, 3, 7, 8]
    w_idx = [8, 11]
    
    add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk)
    
    # Block 9: A2_top * B_top_right (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    u_idx = [8, 9]
    v_idx = [4, 9]
    w_idx = [14]
    
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
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
