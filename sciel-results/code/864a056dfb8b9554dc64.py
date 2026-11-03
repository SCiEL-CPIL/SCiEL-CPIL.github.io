import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Strassen 2x2x2 decomposition
    # U, V, W are 4x7 matrices
    # A, B, C are 2x2 matrices flattened to 4-vectors
    # C = A * B
    # c00 = m0 + m3 - m4 + m6
    # c01 = m2 + m4
    # c10 = m1 + m3
    # c11 = m0 - m1 + m2 + m5
    
    # m0 = (a00+a11)(b00+b11)
    # m1 = (a10+a11)b00
    # m2 = a00(b01-b11)
    # m3 = a11(b10-b00)
    # m4 = (a00+a01)b11
    # m5 = (a10-a00)(b00+b01)
    # m6 = (a01-a11)(b10+b11)
    
    U2 = np.array([
        [1, 0, 0, 1],  # m0
        [0, 0, 1, 1],  # m1
        [1, 0, 0, 0],  # m2
        [0, 0, 0, 1],  # m3
        [1, 1, 0, 0],  # m4
        [1, 0, 1, 0],  # m5
        [0, 1, 0, 1]   # m6
    ]).T # 4x7
    
    V2 = np.array([
        [1, 0, 0, 1],  # m0
        [1, 0, 0, 0],  # m1
        [0, 1, 0, 1],  # m2
        [0, 0, 1, 1],  # m3
        [0, 0, 0, 1],  # m4
        [1, 1, 0, 0],  # m5
        [0, 0, 1, 1]   # m6
    ]).T # 4x7
    
    W2 = np.array([
        [1, 0, 0, 1],  # m0: c00, c11
        [0, 0, 1, 0],  # m1: c10
        [0, 1, 0, 1],  # m2: c01, c11
        [1, 0, 1, 0],  # m3: c00, c10
        [-1, 1, 0, 0], # m4: c00, c01
        [0, 0, 0, 1],  # m5: c11
        [1, 0, 0, 0]   # m6: c00
    ]).T # 4x7
    
    # Check W2:
    # c00: m0(1) + m3(1) - m4(1) + m6(1) -> 1+1-1+1 = 2? No.
    # c00 = m0 + m3 - m4 + m6.
    # W2[0,0]=1, W2[0,3]=1, W2[0,4]=-1, W2[0,6]=1.
    # Sum: 1+1-1+1 = 2. This is wrong.
    # Let's re-verify Strassen formulas.
    # m0 = (a00+a11)(b00+b11)
    # m1 = (a10+a11)b00
    # m2 = a00(b01-b11)
    # m3 = a11(b10-b00)
    # m4 = (a00+a01)b11
    # m5 = (a10-a00)(b00+b01)
    # m6 = (a01-a11)(b10+b11)
    
    # c00 = m0 + m3 - m4 + m6
    # c01 = m2 + m4
    # c10 = m1 + m3
    # c11 = m0 - m1 + m2 + m5
    
    # Let's re-calculate W2 rows (for each c element).
    # c00: m0=1, m3=1, m4=-1, m6=1. Others 0.
    # c01: m2=1, m4=1.
    # c10: m1=1, m3=1.
    # c11: m0=1, m1=-1, m2=1, m5=1.
    
    W2_correct = np.array([
        [1, 0, 0, 1, -1, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, -1, 1, 0, 0, 1, 0]  # c11
    ]).T # 4x7
    
    # Define outer Strassen products
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)B00
    # P2 = A00(B01-B11)
    # P3 = A11(B10-B00)
    # P4 = (A00+A01)B11
    # P5 = (A10-A00)(B00+B01)
    # P6 = (A01-A11)(B10+B11)
    
    # Coefficients for A blocks (A00, A01, A10, A11)
    # Each block is 2x2.
    # We need to map from global A indices to block indices.
    # A is 4x4. Index i*4+j.
    # Block 00: i in 0,1, j in 0,1. Indices: 0,1,4,5.
    # Block 01: i in 0,1, j in 2,3. Indices: 2,3,6,7.
    # Block 10: i in 2,3, j in 0,1. Indices: 8,9,12,13.
    # Block 11: i in 2,3, j in 2,3. Indices: 10,11,14,15.
    
    # For each outer product s, we define L_A_s (4x16) and L_B_s (4x16)
    # L_A_s maps vec(A) to vec(X_s) where X_s is the 2x2 matrix (flattened)
    # Actually, X_s is a 2x2 matrix of blocks? No, X_s is a 2x2 matrix.
    # X_s elements are linear combos of A blocks.
    # So X_s is a 2x2 matrix. Its elements are scalars? No, A blocks are 2x2 matrices.
    # So X_s is a 2x2 matrix where each entry is a 2x2 matrix?
    # No, Strassen operates on blocks.
    # $P_s = X_s Y_s$ where $X_s, Y_s$ are 2x2 matrices.
    # $X_s$ is formed by adding/subtracting blocks of A.
    # So $X_s$ is a 2x2 matrix. Each element of $X_s$ is a 2x2 block of A (or sum of them).
    # So $X_s$ is effectively a 4x4 matrix?
    # Yes, $X_s$ is a 4x4 matrix (concatenation of the 2x2 blocks).
    # But we treat it as a 2x2 matrix of 2x2 blocks for the inner Strassen.
    # The inner Strassen multiplies two 2x2 matrices.
    # So we need the "block values" of $X_s$ and $Y_s$.
    
    # Let's define the block coefficients for each outer product.
    # Blocks: 0:00, 1:01, 2:10, 3:11.
    # P0: A: [1, 0, 0, 1], B: [1, 0, 0, 1]
    # P1: A: [0, 0, 1, 1], B: [1, 0, 0, 0]
    # P2: A: [1, 0, 0, 0], B: [0, 1, 0, -1]
    # P3: A: [0, 0, 0, 1], B: [0, 0, 1, -1]
    # P4: A: [1, 1, 0, 0], B: [0, 0, 0, 1]
    # P5: A: [0, 0, 1, -1], B: [1, 1, 0, 0]
    # P6: A: [0, 1, 0, -1], B: [0, 0, 1, 1]
    
    outer_A = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]) # 7x4
    
    outer_B = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]) # 7x4
    
    # Output coefficients for C blocks
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    outer_C = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]) # 4x7
    
    # Construct global U, V, W
    rank = 49 + 16
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    
    col = 0
    for s in range(7):
        for t in range(7):
            # Inner Strassen term t
            u_t = U2[:, t] # 4-vector
            v_t = V2[:, t] # 4-vector
            w_t = W2_correct[:, t] # 4-vector
            
            # X_s is a 2x2 matrix of blocks.
            # X_s[i] (i=0..3) is the i-th block of X_s (flattened 2x2 matrix).
            # X_s_block_k = sum_b outer_A[s, b] * A_block_b
            # A_block_b is a 2x2 matrix (4 elements).
            # So X_s_block_k is a 2x2 matrix.
            # We need to map this to the 4 elements of the inner X matrix.
            # Inner X is a 2x2 matrix. Its elements are the "scalar" values?
            # No, the inner Strassen multiplies two 2x2 matrices.
            # The "matrices" are the blocks themselves.
            # So the inner X is a 2x2 matrix where each entry is a 2x2 block.
            # This is a 4x4 matrix.
            # But the inner Strassen U2, V2, W2 are for 2x2x2.
            # This implies the inner Strassen treats the blocks as scalars?
            # No. The rank-49 decomposition of 4x4x4 comes from 7 products of 2x2 matrices.
            # Each 2x2 matrix product has rank 7.
            # So we decompose each 2x2 matrix product using Strassen.
            # The "matrices" in the inner Strassen are the 2x2 blocks.
            # So the inner X is a 2x2 matrix. Its elements are the 2x2 blocks of X_s.
            # Wait, X_s IS a 2x2 matrix of blocks.
            # So X_s has 4 elements, each being a 2x2 block.
            # Let's call these elements X_s[0], X_s[1], X_s[2], X_s[3] (row-major).
            # X_s[0] = outer_A[s,0]*A00 + outer_A[s,1]*A01 + ... (2x2 matrix)
            # The inner Strassen U2 has 4 entries corresponding to the 4 elements of a 2x2 matrix.
            # So U2[0] corresponds to element 0 of the inner matrix.
            # So for the global U vector:
            # For each block element k in 0..3 of X_s:
            #   The contribution to A is sum_b outer_A[s,b] * A_block_b.
            #   The weight is U2[k, t].
            #   So U_global += U2[k,t] * (sum_b outer_A[s,b] * A_block_b_vec)
            
            # A_block_b_vec is a 16-vector with 1s at the positions of block b.
            # Block 0: indices 0,1,4,5
            # Block 1: indices 2,3,6,7
            # Block 2: indices 8,9,12,13
            # Block 3: indices 10,11,14,15
            
            block_indices = [
                [0, 1, 4, 5],
                [2, 3, 6, 7],
                [8, 9, 12, 13],
                [10, 11, 14, 15]
            ]
            
            for k in range(4):
                coeff = U2[k, t]
                if coeff != 0:
                    for b in range(4):
                        coeff_b = outer_A[s, b]
                        if coeff_b != 0:
                            for idx in block_indices[b]:
                                U[idx, col] += coeff * coeff_b
            
            # Similarly for V
            # Y_s[k] = sum_b outer_B[s,b] * B_block_b
            # B blocks for first 4 columns.
            # B is 4x5. First 4 cols are 4x4.
            # Block 0: B[0:2, 0:2] -> indices 0,1, 4,5 (row*5+col)
            # Block 1: B[0:2, 2:4] -> indices 2,3, 6,7
            # Block 2: B[2:4, 0:2] -> indices 10,11, 14,15
            # Block 3: B[2:4, 2:4] -> indices 12,13, 16,17
            
            block_indices_B = [
                [0, 1, 5, 6],
                [2, 3, 7, 8],
                [10, 11, 15, 16],
                [12, 13, 17, 18]
            ]
            
            for k in range(4):
                coeff = V2[k, t]
                if coeff != 0:
                    for b in range(4):
                        coeff_b = outer_B[s, b]
                        if coeff_b != 0:
                            for idx in block_indices_B[b]:
                                V[idx, col] += coeff * coeff_b
            
            # W
            # C_block_ij gets contribution alpha_{s, ij} * P_s
            # P_s is the result of inner Strassen.
            # P_s[k,l] = sum_t W2_correct[k,l,t] * M_{s,t}
            # So for fixed s,t, contribution to P_s[k,l] is W2_correct[k,l,t] * M_{s,t}.
            # This P_s contributes to C blocks.
            # C_block_ij[k,l] += outer_C[i,j,s] * P_s[k,l]
            # C_block_ij is a 2x2 matrix.
            # C_block_ij[k,l] corresponds to C element at row i*2+k, col j*2+l.
            # Global C index: (i*2+k)*4 + (j*2+l) for first 4 cols.
            
            for i in range(2):
                for j in range(2):
                    alpha = outer_C[i, j, s]
                    if alpha != 0:
                        for k in range(2):
                            for l in range(2):
                                w_val = W2_correct[k, l, t]
                                if w_val != 0:
                                    c_row = i * 2 + k
                                    c_col = j * 2 + l
                                    c_idx = c_row * 4 + c_col
                                    W[c_idx, col] += alpha * w_val
            
            col += 1
    
    # Add schoolbook for 5th column
    # A (4x4) * B_2 (4x1) = C_2 (4x1)
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1 # C index for col 4, row i
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
