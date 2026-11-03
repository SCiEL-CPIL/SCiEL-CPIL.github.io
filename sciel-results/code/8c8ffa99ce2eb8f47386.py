import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will use the rank-49 decomposition for 4x4x4 twice.
    # Once for B[:, 0:4] and once for B[:, 4:7] (padded).
    # Total rank 98.
    
    def get_rank49_decomposition():
        # Returns U, V, W for 4x4x4 multiplication with rank 49.
        # U: (16, 49), V: (16, 49), W: (16, 49)
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Strassen products M1..M7
        # Each Ms is a 2x2 matrix product.
        # We need to define the linear forms for Xs and Ys.
        # Xs is a 2x2 matrix of linear forms in A.
        # Ys is a 2x2 matrix of linear forms in B.
        
        # Let's define the blocks of A and B.
        # A is 4x4. Blocks A11, A12, A21, A22 are 2x2.
        # Indices in A (flattened 0-15):
        # A11: 0,1,4,5
        # A12: 2,3,6,7
        # A21: 8,9,12,13
        # A22: 10,11,14,15
        
        # B is 4x4. Blocks B11, B12, B21, B22 are 2x2.
        # Indices in B (flattened 0-15):
        # B11: 0,1,4,5
        # B12: 2,3,6,7
        # B21: 8,9,12,13
        # B22: 10,11,14,15
        
        # C is 4x4. Blocks C11, C12, C21, C22 are 2x2.
        # Indices in C (flattened 0-15):
        # C11: 0,1,4,5
        # C12: 2,3,6,7
        # C21: 8,9,12,13
        # C22: 10,11,14,15
        
        # Rank-7 decomposition for 2x2x2:
        # u_k, v_k, w_k are vectors of length 4.
        # Mapping local index (0-3) to global block index.
        
        strassen_u = [
            [1, 0, 0, 1], # 1
            [0, 1, 0, 1], # 2
            [1, 0, 0, 0], # 3
            [0, 0, 0, 1], # 4
            [1, 1, 0, 0], # 5
            [-1, 0, 1, 0],# 6
            [0, 1, 0, -1] # 7
        ]
        strassen_v = [
            [1, 0, 0, 1], # 1
            [1, 0, 0, 0], # 2
            [0, 1, 0, -1],# 3
            [0, 1, 0, -1],# 4
            [0, 0, 0, 1], # 5
            [1, 1, 0, 0], # 6
            [0, 1, 0, 1]  # 7
        ]
        strassen_w = [
            [1, 1, 1, 1], # 1
            [0, 0, 1, 1], # 2
            [1, 0, 0, 0], # 3
            [0, 0, 0, 1], # 4
            [0, 1, 0, 0], # 5
            [0, 0, 1, 0], # 6
            [0, 0, 0, 1]  # 7
        ]
        
        # Define Xs and Ys for s=1..7
        # Xs and Ys are 2x2 matrices.
        # Each entry is a vector of length 16 (coefficients for A or B).
        
        # Helper to get block indices
        def block_indices(block_type, block_id):
            # block_type: 'A' or 'B'
            # block_id: 0 (11), 1 (12), 2 (21), 3 (22)
            if block_id == 0: # 11
                return [0, 1, 4, 5]
            elif block_id == 1: # 12
                return [2, 3, 6, 7]
            elif block_id == 2: # 21
                return [8, 9, 12, 13]
            else: # 22
                return [10, 11, 14, 15]
        
        # Xs, Ys definitions
        # s=1: X1 = A11+A22, Y1 = B11+B22
        # X1 is a 2x2 matrix? No, X1 is the matrix (A11+A22).
        # So X1[0,0] = A11[0,0] + A22[0,0], etc.
        # We represent Xs as a 4x4 matrix of coefficients?
        # No, Xs is a 2x2 matrix where each entry is a linear form.
        # Let's represent Xs as a 4-vector of 16-vectors?
        # Or just a 4x4 matrix of coefficients for the 16 inputs.
        
        # Let's build U, V, W directly.
        # For each s in 1..7, and each k in 1..7:
        # We have a term u_k \otimes v_k \otimes w_k for the local 2x2x2.
        # This term corresponds to:
        # (sum_local_i u_k[i] Xs[i]) * (sum_local_j v_k[j] Ys[j]) * (sum_local_l w_k[l] C_block[l])
        
        # We need to map local indices to global indices.
        
        for s in range(7):
            # Define Xs and Ys
            # Xs is 2x2. Xs[i,j] is a linear form in A.
            # Ys is 2x2. Ys[i,j] is a linear form in B.
            
            Xs = np.zeros((2, 2, 16))
            Ys = np.zeros((2, 2, 16))
            
            if s == 0: # M1
                # X1 = A11 + A22
                # X1[0,0] = A11[0,0] + A22[0,0] -> indices 0, 10
                # X1[0,1] = A11[0,1] + A22[0,1] -> indices 1, 11
                # X1[1,0] = A11[1,0] + A22[1,0] -> indices 4, 14
                # X1[1,1] = A11[1,1] + A22[1,1] -> indices 5, 15
                Xs[0,0,0] = 1; Xs[0,0,10] = 1
                Xs[0,1,1] = 1; Xs[0,1,11] = 1
                Xs[1,0,4] = 1; Xs[1,0,14] = 1
                Xs[1,1,5] = 1; Xs[1,1,15] = 1
                
                Ys[0,0,0] = 1; Ys[0,0,10] = 1
                Ys[0,1,1] = 1; Ys[0,1,11] = 1
                Ys[1,0,4] = 1; Ys[1,0,14] = 1
                Ys[1,1,5] = 1; Ys[1,1,15] = 1
                
            elif s == 1: # M2
                # X2 = A21 + A22
                # X2[0,0] = A21[0,0] + A22[0,0] -> indices 8, 10
                # X2[0,1] = A21[0,1] + A22[0,1] -> indices 9, 11
                # X2[1,0] = A21[1,0] + A22[1,0] -> indices 12, 14
                # X2[1,1] = A21[1,1] + A22[1,1] -> indices 13, 15
                Xs[0,0,8] = 1; Xs[0,0,10] = 1
                Xs[0,1,9] = 1; Xs[0,1,11] = 1
                Xs[1,0,12] = 1; Xs[1,0,14] = 1
                Xs[1,1,13] = 1; Xs[1,1,15] = 1
                
                # Y2 = B11
                # Y2[0,0] = B11[0,0] -> 0
                # Y2[0,1] = B11[0,1] -> 1
                # Y2[1,0] = B11[1,0] -> 4
                # Y2[1,1] = B11[1,1] -> 5
                Ys[0,0,0] = 1
                Ys[0,1,1] = 1
                Ys[1,0,4] = 1
                Ys[1,1,5] = 1
                
            elif s == 2: # M3
                # X3 = A11
                Xs[0,0,0] = 1
                Xs[0,1,1] = 1
                Xs[1,0,4] = 1
                Xs[1,1,5] = 1
                
                # Y3 = B12 - B22
                # Y3[0,0] = B12[0,0] - B22[0,0] -> 2 - 10
                # Y3[0,1] = B12[0,1] - B22[0,1] -> 3 - 11
                # Y3[1,0] = B12[1,0] - B22[1,0] -> 6 - 14
                # Y3[1,1] = B12[1,1] - B22[1,1] -> 7 - 15
                Ys[0,0,2] = 1; Ys[0,0,10] = -1
                Ys[0,1,3] = 1; Ys[0,1,11] = -1
                Ys[1,0,6] = 1; Ys[1,0,14] = -1
                Ys[1,1,7] = 1; Ys[1,1,15] = -1
                
            elif s == 3: # M4
                # X4 = A22
                Xs[0,0,10] = 1
                Xs[0,1,11] = 1
                Xs[1,0,14] = 1
                Xs[1,1,15] = 1
                
                # Y4 = B21 - B11
                # Y4[0,0] = B21[0,0] - B11[0,0] -> 8 - 0
                # Y4[0,1] = B21[0,1] - B11[0,1] -> 9 - 1
                # Y4[1,0] = B21[1,0] - B11[1,0] -> 12 - 4
                # Y4[1,1] = B21[1,1] - B11[1,1] -> 13 - 5
                Ys[0,0,8] = 1; Ys[0,0,0] = -1
                Ys[0,1,9] = 1; Ys[0,1,1] = -1
                Ys[1,0,12] = 1; Ys[1,0,4] = -1
                Ys[1,1,13] = 1; Ys[1,1,5] = -1
                
            elif s == 4: # M5
                # X5 = A11 + A12
                # X5[0,0] = A11[0,0] + A12[0,0] -> 0 + 2
                # X5[0,1] = A11[0,1] + A12[0,1] -> 1 + 3
                # X5[1,0] = A11[1,0] + A12[1,0] -> 4 + 6
                # X5[1,1] = A11[1,1] + A12[1,1] -> 5 + 7
                Xs[0,0,0] = 1; Xs[0,0,2] = 1
                Xs[0,1,1] = 1; Xs[0,1,3] = 1
                Xs[1,0,4] = 1; Xs[1,0,6] = 1
                Xs[1,1,5] = 1; Xs[1,1,7] = 1
                
                # Y5 = B22
                Ys[0,0,10] = 1
                Ys[0,1,11] = 1
                Ys[1,0,14] = 1
                Ys[1,1,15] = 1
                
            elif s == 5: # M6
                # X6 = A21 - A11
                # X6[0,0] = A21[0,0] - A11[0,0] -> 8 - 0
                # X6[0,1] = A21[0,1] - A11[0,1] -> 9 - 1
                # X6[1,0] = A21[1,0] - A11[1,0] -> 12 - 4
                # X6[1,1] = A21[1,1] - A11[1,1] -> 13 - 5
                Xs[0,0,8] = 1; Xs[0,0,0] = -1
                Xs[0,1,9] = 1; Xs[0,1,1] = -1
                Xs[1,0,12] = 1; Xs[1,0,4] = -1
                Xs[1,1,13] = 1; Xs[1,1,5] = -1
                
                # Y6 = B11 + B12
                # Y6[0,0] = B11[0,0] + B12[0,0] -> 0 + 2
                # Y6[0,1] = B11[0,1] + B12[0,1] -> 1 + 3
                # Y6[1,0] = B11[1,0] + B12[1,0] -> 4 + 6
                # Y6[1,1] = B11[1,1] + B12[1,1] -> 5 + 7
                Ys[0,0,0] = 1; Ys[0,0,2] = 1
                Ys[0,1,1] = 1; Ys[0,1,3] = 1
                Ys[1,0,4] = 1; Ys[1,0,6] = 1
                Ys[1,1,5] = 1; Ys[1,1,7] = 1
                
            elif s == 6: # M7
                # X7 = A12 - A22
                # X7[0,0] = A12[0,0] - A22[0,0] -> 2 - 10
                # X7[0,1] = A12[0,1] - A22[0,1] -> 3 - 11
                # X7[1,0] = A12[1,0] - A22[1,0] -> 6 - 14
                # X7[1,1] = A12[1,1] - A22[1,1] -> 7 - 15
                Xs[0,0,2] = 1; Xs[0,0,10] = -1
                Xs[0,1,3] = 1; Xs[0,1,11] = -1
                Xs[1,0,6] = 1; Xs[1,0,14] = -1
                Xs[1,1,7] = 1; Xs[1,1,15] = -1
                
                # Y7 = B21 + B22
                # Y7[0,0] = B21[0,0] + B22[0,0] -> 8 + 10
                # Y7[0,1] = B21[0,1] + B22[0,1] -> 9 + 11
                # Y7[1,0] = B21[1,0] + B22[1,0] -> 12 + 14
                # Y7[1,1] = B21[1,1] + B22[1,1] -> 13 + 15
                Ys[0,0,8] = 1; Ys[0,0,10] = 1
                Ys[0,1,9] = 1; Ys[0,1,11] = 1
                Ys[1,0,12] = 1; Ys[1,0,14] = 1
                Ys[1,1,13] = 1; Ys[1,1,15] = 1
            
            # Now add the 7 terms for this s
            for k in range(7):
                col = s * 7 + k
                # u_k is vector of length 4
                # v_k is vector of length 4
                # w_k is vector of length 4
                
                # For each local index i (0-3) for U
                for i in range(4):
                    if strassen_u[k][i] != 0:
                        # i corresponds to Xs[i//2, i%2]
                        xi, xj = i // 2, i % 2
                        # Add strassen_u[k][i] * Xs[xi, xj, :] to U[:, col]
                        U[:, col] += strassen_u[k][i] * Xs[xi, xj, :]
                
                # For each local index j (0-3) for V
                for j in range(4):
                    if strassen_v[k][j] != 0:
                        # j corresponds to Ys[j//2, j%2]
                        yj1, yj2 = j // 2, j % 2
                        # Add strassen_v[k][j] * Ys[yj1, yj2, :] to V[:, col]
                        V[:, col] += strassen_v[k][j] * Ys[yj1, yj2, :]
                
                # For each local index l (0-3) for W
                for l in range(4):
                    if strassen_w[k][l] != 0:
                        # l corresponds to C block l
                        # C block 0: C11, 1: C12, 2: C21, 3: C22
                        # Indices in C:
                        # C11: 0,1,4,5
                        # C12: 2,3,6,7
                        # C21: 8,9,12,13
                        # C22: 10,11,14,15
                        c_block = l
                        if c_block == 0:
                            inds = [0, 1, 4, 5]
                        elif c_block == 1:
                            inds = [2, 3, 6, 7]
                        elif c_block == 2:
                            inds = [8, 9, 12, 13]
                        else:
                            inds = [10, 11, 14, 15]
                        # Add strassen_w[k][l] to W[inds, col]
                        for idx in inds:
                            W[idx, col] += strassen_w[k][l]
                            
        return U, V, W

    U49, V49, W49 = get_rank49_decomposition()
    
    # Now construct for 4x4x7
    # U: (16, 98)
    # V: (28, 98)
    # W: (28, 98)
    
    U = np.zeros((16, 98))
    V = np.zeros((28, 98))
    W = np.zeros((28, 98))
    
    # First 49 columns: B[:, 0:4]
    U[:, 0:49] = U49
    V[0:16, 0:49] = V49
    # B has 28 rows. First 16 correspond to B[:, 0:4].
    # W: C[:, 0:4]
    # W is (p*n, rank) = (28, 98)
    # W[k*n+i, c]
    # For the first part, k is 0..3, i is 0..3.
    # W[0:16, 0:49] = W49
    
    # Second 49 columns: B[:, 4:7]
    # B[:, 4:7] is 4x3. Pad to 4x4 with zeros.
    # U is same: U[:, 49:98] = U49
    U[:, 49:98] = U49
    # V: B[:, 4:7] is rows 16..22 of B.
    # V49 is (16, 49).
    # We need to map V49 to V[16:20, 49:98]?
    # B is 4x7. Flattened: 0..27.
    # B[:, 0:4] is 0..15.
    # B[:, 4:7] is 16..22.
    # B[:, 7:8] (padded) is 23..27? No, just zeros.
    # V49 corresponds to B blocks B11, B12, B21, B22.
    # B11: 0..3 (cols 0,1), B12: 4..7 (cols 2,3)
    # B21: 8..11 (cols 0,1), B22: 12..15 (cols 2,3)
    # For B[:, 4:7], the columns are 4,5,6.
    # We pad column 7 with 0.
    # So B_padded has columns 4,5,6,7.
    # B_padded[:, 0] = B[:, 4] -> global index 16
    # B_padded[:, 1] = B[:, 5] -> global index 17
    # B_padded[:, 2] = B[:, 6] -> global index 18
    # B_padded[:, 3] = 0 -> global index 19? No, we don't have index 19.
    # We just leave V[19:28, 49:98] as 0.
    
    # Mapping V49 indices to global V indices:
    # V49[0] -> B_padded[0,0] -> B[4,0] -> global 16
    # V49[1] -> B_padded[0,1] -> B[4,1] -> global 17
    # V49[2] -> B_padded[0,2] -> B[4,2] -> global 18
    # V49[3] -> B_padded[0,3] -> 0
    # V49[4] -> B_padded[1,0] -> B[4,4] -> global 20?
    # B is 4x7. Flattened row-major.
    # B[i,j] -> i*7 + j.
    # B[0,4] -> 4
    # B[0,5] -> 5
    # B[0,6] -> 6
    # B[1,4] -> 1*7+4 = 11
    # B[1,5] -> 12
    # B[1,6] -> 13
    # B[2,4] -> 2*7+4 = 18
    # B[2,5] -> 19
    # B[2,6] -> 20
    # B[3,4] -> 3*7+4 = 25
    # B[3,5] -> 26
    # B[3,6] -> 27
    
    # V49 indices 0..15 correspond to B_padded blocks.
    # B_padded is 4x4.
    # B_padded[0,0] -> B[0,4] -> 4
    # B_padded[0,1] -> B[0,5] -> 5
    # B_padded[0,2] -> B[0,6] -> 6
    # B_padded[0,3] -> 0
    # B_padded[1,0] -> B[1,4] -> 11
    # B_padded[1,1] -> B[1,5] -> 12
    # B_padded[1,2] -> B[1,6] -> 13
    # B_padded[1,3] -> 0
    # B_padded[2,0] -> B[2,4] -> 18
    # B_padded[2,1] -> B[2,5] -> 19
    # B_padded[2,2] -> B[2,6] -> 20
    # B_padded[2,3] -> 0
    # B_padded[3,0] -> B[3,4] -> 25
    # B_padded[3,1] -> B[3,5] -> 26
    # B_padded[3,2] -> B[3,6] -> 27
    # B_padded[3,3] -> 0
    
    # Create mapping
    v_map = [4, 5, 6, -1, 11, 12, 13, -1, 18, 19, 20, -1, 25, 26, 27, -1]
    
    for i in range(16):
        if v_map[i] != -1:
            V[v_map[i], 49:98] = V49[i, :]
            
    # W for the second part: C[:, 4:7]
    # C is 4x7.
    # C[:, 4:7] is columns 4,5,6.
    # W49 corresponds to C_padded blocks C11, C12, C21, C22.
    # C_padded is 4x4.
    # C_padded[:, 0] = C[:, 4]
    # C_padded[:, 1] = C[:, 5]
    # C_padded[:, 2] = C[:, 6]
    # C_padded[:, 3] = 0
    
    # W49 indices 0..15 correspond to C_padded.
    # C_padded[0,0] -> C[0,4] -> 4
    # C_padded[0,1] -> C[0,5] -> 5
    # C_padded[0,2] -> C[0,6] -> 6
    # C_padded[0,3] -> 0
    # C_padded[1,0] -> C[1,4] -> 11
    # C_padded[1,1] -> C[1,5] -> 12
    # C_padded[1,2] -> C[1,6] -> 13
    # C_padded[1,3] -> 0
    # C_padded[2,0] -> C[2,4] -> 18
    # C_padded[2,1] -> C[2,5] -> 19
    # C_padded[2,2] -> C[2,6] -> 20
    # C_padded[2,3] -> 0
    # C_padded[3,0] -> C[3,4] -> 25
    # C_padded[3,1] -> C[3,5] -> 26
    # C_padded[3,2] -> C[3,6] -> 27
    # C_padded[3,3] -> 0
    
    w_map = [4, 5, 6, -1, 11, 12, 13, -1, 18, 19, 20, -1, 25, 26, 27, -1]
    
    for i in range(16):
        if w_map[i] != -1:
            W[w_map[i], 49:98] = W49[i, :]
            
    # Convert to list of lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
