import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create schoolbook decomposition
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Helper to create Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        # Indices for U (A): 0:11, 1:12, 2:21, 3:22
        # Indices for V (B): 0:11, 1:12, 2:21, 3:22
        # Indices for W (C): 0:11, 1:12, 2:21, 3:22
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C11 += M1, C22 += M1
        
        # M2 = (A11+A12)B22
        U[0, 1] = 1; U[1, 1] = 1
        V[3, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1 # C21 += M2, C22 += M2
        
        # M3 = A11(B22-B21)
        U[0, 2] = 1
        V[3, 2] = 1; V[2, 2] = -1
        W[1, 2] = 1; W[3, 2] = -1 # C12 += M3, C22 -= M3
        
        # M4 = A22(B11-B12)
        U[3, 3] = 1
        V[0, 3] = 1; V[1, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1 # C11 += M4, C21 += M4
        
        # M5 = (A21+A22)B11
        U[2, 4] = 1; U[3, 4] = 1
        V[0, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1 # C11 -= M5, C12 += M5
        
        # M6 = (A12-A22)(B21+B22)
        U[1, 5] = 1; U[3, 5] = -1
        V[2, 5] = 1; V[3, 5] = 1
        W[3, 5] = 1 # C22 += M6
        
        # M7 = (A11-A21)(B12+B21)
        U[0, 6] = 1; U[2, 6] = -1
        V[1, 6] = 1; V[2, 6] = 1
        W[0, 6] = 1 # C11 += M7
        
        return U, V, W

    # Build 4x4x4 decomposition using nested Strassen
    def build_4x4x4():
        # 7 block multiplications, each is 2x2x2 Strassen
        # Total rank 49
        U_blocks = []
        V_blocks = []
        W_blocks = []
        
        # Get the 2x2x2 Strassen matrices
        U_s, V_s, W_s = strassen_2x2x2()
        
        # We need to map the 4x4x4 tensor to 7 "super" multiplications
        # The super multiplications are defined by the Strassen coefficients on the block level.
        # Block indices: 0,1,2,3 correspond to (1,1), (1,2), (2,1), (2,2)
        # Let's define the linear combinations for A, B, C blocks.
        
        # A blocks: A00, A01, A10, A11 (using 0-indexed block coords)
        # B blocks: B00, B01, B10, B11
        # C blocks: C00, C01, C10, C11
        
        # Strassen formulas for blocks:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A00+A01)B11
        # M3 = A00(B11-B10)
        # M4 = A11(B00-B01)
        # M5 = (A10+A11)B00
        # M6 = (A01-A11)(B10+B11)
        # M7 = (A00-A10)(B01+B10)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 + M6
        
        # We will iterate over the 7 M's.
        # For each M, we have a linear combination of A-blocks and B-blocks.
        # Each A-block is a 2x2 matrix, so 4 scalars.
        # Each B-block is a 2x2 matrix, so 4 scalars.
        # The product M is a 2x2 matrix.
        # We compute this product using Strassen 2x2x2 on the scalars.
        
        # The "super" Strassen coefficients give us the weights for the blocks.
        # Let's define the block weights for each M.
        
        # M1: A: [1, 0, 0, 1], B: [1, 0, 0, 1]
        # M2: A: [1, 1, 0, 0], B: [0, 0, 0, 1]
        # M3: A: [1, 0, 0, 0], B: [0, 0, -1, 1]
        # M4: A: [0, 0, 0, 1], B: [1, -1, 0, 0]
        # M5: A: [0, 0, 1, 1], B: [1, 0, 0, 0]
        # M6: A: [0, 1, 0, -1], B: [0, 0, 1, 1]
        # M7: A: [1, 0, -1, 0], B: [0, 1, 1, 0]
        
        # C weights:
        # C00: [1, 0, 0, 1, -1, 0, 1] (M1, M2, M3, M4, M5, M6, M7)
        # C01: [0, 0, 1, 0, 1, 0, 0]
        # C10: [0, 1, 0, 1, 0, 0, 0]
        # C11: [1, 1, -1, 0, 0, 1, 0]
        
        A_weights = [
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 0, -1],
            [1, 0, -1, 0]
        ]
        B_weights = [
            [1, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, -1, 1],
            [1, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 1, 0]
        ]
        C_weights = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, 1, -1, 0, 0, 1, 0]
        ]
        
        U_4 = np.zeros((16, 49))
        V_4 = np.zeros((16, 49))
        W_4 = np.zeros((16, 49))
        
        # U_4 rows: A_block * 4 + A_scalar
        # V_4 rows: B_block * 4 + B_scalar
        # W_4 rows: C_block * 4 + C_scalar
        
        for m_idx in range(7):
            # Get the scalar Strassen matrices for this M
            # We use the same U_s, V_s, W_s for all M's, but scaled by block weights?
            # No, the block weights apply to the *blocks*.
            # The scalar Strassen applies to the entries within the blocks.
            
            # The effective A for M_idx is sum(A_weights[m_idx][b] * A_block_b for b in 0..3)
            # This is a 2x2 matrix.
            # The entries are linear combinations of A scalars.
            
            # We can construct the U_4 contributions by combining the block weights and scalar Strassen weights.
            
            # U_s has shape (4, 7). Rows are A_scalars (0,1,2,3), Cols are scalar terms.
            # For a given M_idx, the A_block weights are A_weights[m_idx].
            # The scalar term t contributes to U_4 row (block_b * 4 + scalar_s) with weight A_weights[m_idx][b] * U_s[s, t].
            
            for t in range(7):
                col = m_idx * 7 + t
                # Update U_4
                for b in range(4):
                    w_block = A_weights[m_idx][b]
                    if w_block != 0:
                        for s in range(4):
                            w_scalar = U_s[s, t]
                            if w_scalar != 0:
                                U_4[b * 4 + s, col] += w_block * w_scalar
                                
                # Update V_4
                for b in range(4):
                    w_block = B_weights[m_idx][b]
                    if w_block != 0:
                        for s in range(4):
                            w_scalar = V_s[s, t]
                            if w_scalar != 0:
                                V_4[b * 4 + s, col] += w_block * w_scalar
                                
                # Update W_4
                for c_block in range(4):
                    w_c_block = C_weights[c_block][m_idx]
                    if w_c_block != 0:
                        for s in range(4):
                            w_scalar = W_s[s, t]
                            if w_scalar != 0:
                                W_4[c_block * 4 + s, col] += w_c_block * w_scalar
                                
        return U_4, V_4, W_4

    # Combine the parts
    # Part 1: 4x4x4 (Rank 49)
    U1, V1, W1 = build_4x4x4()
    
    # Part 2: 4x4x2 (Rank 32)
    U2, V2, W2 = schoolbook(4, 4, 2)
    
    # Part 3: 4x1x6 (Rank 24)
    U3, V3, W3 = schoolbook(4, 1, 6)
    
    # Global dimensions
    rank = 49 + 32 + 24
    U = np.zeros((20, rank))
    V = np.zeros((30, rank))
    W = np.zeros((24, rank))
    
    # Map Part 1 (4x4x4)
    # i in 0..3, j in 0..3, k in 0..3
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, 0:49] = U1[i * 4 + j, :]
    for j in range(4):
        for k in range(4):
            V[j * 6 + k, 0:49] = V1[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W1[k * 4 + i, :]
            
    # Map Part 2 (4x4x2) -> k in 4..5
    # i in 0..3, j in 0..3, k in 0..1 (local)
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, 49:81] = U2[i * 4 + j, :]
    for j in range(4):
        for k in range(2):
            V[j * 6 + (k + 4), 49:81] = V2[j * 4 + k, :]
    for k in range(2):
        for i in range(4):
            W[(k + 4) * 4 + i, 49:81] = W2[k * 4 + i, :]
            
    # Map Part 3 (4x1x6) -> j in 4..4
    # i in 0..3, j in 0..0 (local), k in 0..5
    for i in range(4):
        U[i * 5 + 4, 81:105] = U3[i * 1 + 0, :]
    for k in range(6):
        V[4 * 6 + k, 81:105] = V3[0 * 6 + k, :]
    for k in range(6):
        for i in range(4):
            W[k * 4 + i, 81:105] = W3[k * 4 + i, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
