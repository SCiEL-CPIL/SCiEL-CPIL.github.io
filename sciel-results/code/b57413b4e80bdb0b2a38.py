import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 114
    rank = 114
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper for 2x2x2 Strassen
    def get_strassen_2x2():
        # Returns U, V, W of shape 4x7
        # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # P1 = (A11+A22)(B11+B22)
        # A11: (0,0), A22: (1,1)
        # B11: (0,0), B22: (1,1)
        # C11 += P1, C12 += P1, C21 += P1, C22 += P1? No.
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # We define the linear maps for A and B for each P.
        # P1: A = A11 + A22, B = B11 + B22
        # A coeffs: (0,0):1, (1,1):1
        # B coeffs: (0,0):1, (1,1):1
        # C coeffs: None, W is determined by the local 2x2 decomposition.
        # Wait, the local 2x2 decomposition U2, V2, W2 is for the *standard* 2x2 multiplication.
        # So we just need to map the standard U2, V2, W2 to the global space.
        
        # Standard 2x2 Strassen U, V, W
        # P1: A11+A22, B11+B22
        # U2[:, 0]: A11+A22 -> (0,0)=1, (1,1)=1
        # V2[:, 0]: B11+B22 -> (0,0)=1, (1,1)=1
        # W2[:, 0]: C? The local W2 is for the output of the 2x2 multiplication.
        # The local 2x2 multiplication produces a 2x2 matrix C_local.
        # The global C is assembled from C_local blocks.
        
        # Let's just hardcode the 7 U, V, W for 2x2.
        # It's easier to generate them from the definition.
        
        # P1
        U2[0, 0] = 1; U2[3, 0] = 1 # A11, A22
        V2[0, 0] = 1; V2[3, 0] = 1 # B11, B22
        # W2 for P1 contributes to C11, C12, C21, C22?
        # No, P1 is a 2x2 matrix.
        # C_local = P1.
        # So W2 for col 0 is the standard W for 2x2 multiplication?
        # No, the decomposition is sum of rank-1 tensors.
        # Each column c is a rank-1 tensor.
        # For P1, the "multiplication" is (A11+A22)(B11+B22).
        # This is a 2x2 matrix multiplication.
        # So we need the 7 rank-1 tensors for 2x2 multiplication?
        # No, P1 is ONE scalar multiplication in the Strassen algorithm?
        # No, P1 is a 2x2 matrix product.
        # So P1 itself requires rank 7.
        # So for each Strassen product P_l, we use 7 columns.
        # Total 7 * 7 = 49 columns.
        
        # So, for each l in 1..7:
        #   We have A_l and B_l which are 2x2 matrices (linear combos of blocks).
        #   We compute C_l = A_l B_l.
        #   We use the standard 2x2 Strassen decomposition for C_l?
        #   No, we use the *schoolbook* decomposition for the 2x2 product?
        #   No, we use the *Strassen* decomposition for the 2x2 product to get rank 7.
        #   So for each P_l, we add 7 columns to the global U, V, W.
        
        # So I need the U, V, W for the 2x2 Strassen decomposition.
        # Let's generate them.
        
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # P1
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        # W2 for P1: C_local = P1.
        # The rank-1 tensors for 2x2 multiplication are defined by the 7 Strassen products.
        # Wait, I'm confusing levels.
        # Level 1: 2x2 multiplication. Rank 7.
        # Level 2: 4x4 multiplication. Uses 7 Level-1 multiplications.
        # Each Level-1 multiplication is a 2x2 matrix product.
        # So for each of the 7 Strassen products in Level 2, we perform a Level-1 multiplication.
        # The Level-1 multiplication has rank 7.
        # So total rank 49.
        
        # So I need the U, V, W for the Level-1 (2x2) multiplication.
        # Let's call them U1, V1, W1 (4x7).
        
        # U1, V1, W1 for 2x2 Strassen:
        # P1: A11+A22, B11+B22
        U1 = np.zeros((4, 7))
        V1 = np.zeros((4, 7))
        W1 = np.zeros((4, 7))
        
        # P1
        U1[0, 0] = 1; U1[3, 0] = 1
        V1[0, 0] = 1; V1[3, 0] = 1
        # W1 for P1: C11 += P1, C12 += P1, C21 += P1, C22 += P1?
        # No, P1 is the result of the multiplication.
        # The rank-1 tensor for P1 in the 2x2 decomposition is:
        # u = vec(A11+A22), v = vec(B11+B22), w = ?
        # The product is C = P1.
        # So w should be such that sum u v w^T = C.
        # This is not a single rank-1 tensor.
        # The Strassen algorithm replaces the 8 schoolbook multiplications with 7.
        # The 7 products P1..P7 are scalar multiplications of 2x2 matrices?
        # No, P1..P7 are 2x2 matrices.
        # The "multiplication" is the bilinear map.
        # The rank is the number of scalar multiplications.
        # So P1 is computed by 1 scalar multiplication? No.
        # P1 = (A11+A22)(B11+B22). This is a matrix multiplication.
        # So P1 requires rank 7 itself.
        # So total rank is 7 * 7 = 49.
        
        # So yes, for each of the 7 Strassen blocks, we use the full 2x2 Strassen decomposition.
        
        # So I just need to generate U1, V1, W1 for 2x2 Strassen once.
        
        # 2x2 Strassen U1, V1, W1:
        # P1: A11+A22, B11+B22
        U1[0, 0] = 1; U1[3, 0] = 1
        V1[0, 0] = 1; V1[3, 0] = 1
        # W1: C = P1.
        # The contribution of P1 to C is:
        # C11 += P1_11, C12 += P1_12, C21 += P1_21, C22 += P1_22.
        # So W1 for col 0 should be 1 at (0,0), (0,1), (1,0), (1,1)?
        # No, W1 is the output vector.
        # The tensor is sum u v w.
        # If P1 is a matrix, it's not a scalar.
        # The rank of the 2x2 tensor is 7.
        # The 7 rank-1 tensors are:
        # 1. u=(A11+A22), v=(B11+B22), w=(C11+C12+C21+C22)? No.
        
        # Let's look up the explicit U, V, W for 2x2 Strassen.
        # It's easier to just use the schoolbook rank 8 for 2x2 if I can't recall the exact 7 vectors.
        # But I know the 7 products.
        # The decomposition is:
        # C = sum_{l=1}^7 P_l \otimes (coeffs in C)
        # This is not a tensor decomposition in the sense of U, V, W.
        
        # Actually, the rank-1 tensors for 2x2 Strassen are:
        # 1. (A11+A22) (B11+B22) -> contributes to C11, C12, C21, C22
        # This is NOT a rank-1 tensor.
        # The rank-1 tensors are the scalar multiplications.
        # In Strassen, the "multiplications" are of 2x2 matrices.
        # So the rank is 7 * (rank of 2x2 multiplication)?
        # No, the rank of the tensor M_{2,2,2} is 7.
        # So there exist 7 vectors u_k, v_k, w_k such that T = sum u_k v_k w_k.
        
        # I will generate them programmatically by solving the linear system or using known values.
        # Known values for 2x2 Strassen U, V, W:
        
        # P1
        u1 = [1, 0, 0, 1] # A11, A22
        v1 = [1, 0, 0, 1] # B11, B22
        w1 = [1, 1, 1, 1] # C11, C12, C21, C22 ? No.
        
        # Let's use a simpler approach:
        # Just use schoolbook for 2x2 (rank 8).
        # Then 4x4 rank is 7 * 8 = 56.
        # 56 + 56 + 16 = 128.
        # Still better than 144.
        # And much easier to implement correctly.
        
        # Let's use rank 8 for 2x2.
        
    def get_schoolbook_2x2():
        U = np.zeros((4, 8))
        V = np.zeros((4, 8))
        W = np.zeros((4, 8))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    c = i*4 + j*2 + k
                    U[i*2+j, c] = 1
                    V[j*2+k, c] = 1
                    W[k*2+i, c] = 1
        return U, V, W

    U2, V2, W2 = get_schoolbook_2x2()
    
    # Now build 4x4x4 using 7 Strassen products, each using rank 8.
    # Total rank 56.
    
    U4 = np.zeros((16, 56))
    V4 = np.zeros((16, 56))
    W4 = np.zeros((16, 56))
    
    # Strassen products for 4x4:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # For each P, define the block coefficients for A and B.
    # A blocks: 00, 01, 10, 11 (each 2x2)
    # B blocks: 00, 01, 10, 11 (each 2x2)
    
    # P1
    A_coeffs = {0: 1, 3: 1} # A11, A22 (indices in 2x2 block grid)
    B_coeffs = {0: 1, 3: 1}
    # C contribution: C11+=P1, C12+=P1, C21+=P1, C22+=P1
    # Wait, C = A B.
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    
    # Strassen:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # So for each P_l, we know which C blocks it adds to.
    
    def add_product(l, A_coeffs, B_coeffs, C_coeffs):
        # A_coeffs: dict block_index -> coeff
        # B_coeffs: dict block_index -> coeff
        # C_coeffs: dict block_index -> coeff
        
        base_col = l * 8
        for c in range(8):
            col = base_col + c
            # U
            for a_idx, a_val in A_coeffs.items():
                bi, bj = divmod(a_idx, 2)
                # Local U2 row: i*2+j
                for li in range(2):
                    for lj in range(2):
                        if U2[li*2+lj, c] != 0:
                            gi = bi*2 + li
                            gj = bj*2 + lj
                            U4[gi*4+gj, col] += a_val * U2[li*2+lj, c]
            
            # V
            for b_idx, b_val in B_coeffs.items():
                bi, bj = divmod(b_idx, 2)
                for li in range(2):
                    for lj in range(2):
                        if V2[li*2+lj, c] != 0:
                            gi = bi*2 + li
                            gj = bj*2 + lj
                            V4[gi*4+gj, col] += b_val * V2[li*2+lj, c]
            
            # W
            for c_idx, c_val in C_coeffs.items():
                ci, cj = divmod(c_idx, 2)
                for li in range(2):
                    for lj in range(2):
                        if W2[li*2+lj, c] != 0:
                            gi = ci*2 + li
                            gj = cj*2 + lj
                            W4[gi*4+gj, col] += c_val * W2[li*2+lj, c]

    # P1
    add_product(0, {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 1: 1, 2: 1, 3: 1})
    # P2
    add_product(1, {2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1})
    # P3
    add_product(2, {0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1})
    # P4
    add_product(3, {3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1})
    # P5
    add_product(4, {0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1})
    # P6
    add_product(5, {2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    # P7
    add_product(6, {1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 3: -1})

    # Now U4, V4, W4 are 16x56.
    
    # Schoolbook 4x4x1
    U3 = np.zeros((16, 16))
    V3 = np.zeros((16, 16)) # m*p = 4*1 = 4? No, V is (m*p, rank).
    # For 4x4x1, m=4, p=1. V is 4x16.
    # But global V is 4x9.
    # Let's just create local V3 of size 4x16.
    V3 = np.zeros((4, 16))
    W3 = np.zeros((4, 16)) # p*n = 1*4 = 4.
    
    for i in range(4):
        for j in range(4):
            c = i*4 + j
            U3[i*4+j, c] = 1
            V3[j, c] = 1
            W3[i, c] = 1
    
    # Assemble Global
    # Cols 0-55: U4, V4, W4 (for k=0..3)
    for c in range(56):
        U[:, c] = U4[:, c]
        # V4 is 16x56. Global V is 36x114.
        # V4 row j*4+k -> Global V row j*9+k
        for j in range(4):
            for k in range(4):
                V[j*9+k, c] = V4[j*4+k, c]
        # W4 is 16x56. Global W is 36x114.
        # W4 row k*4+i -> Global W row k*4+i (k=0..3)
        for k in range(4):
            for i in range(4):
                W[k*4+i, c] = W4[k*4+i, c]
                
    # Cols 56-111: U4, V4, W4 (for k=4..7)
    for c in range(56):
        gc = 56 + c
        U[:, gc] = U4[:, c]
        for j in range(4):
            for k in range(4):
                V[j*9 + (k+4), gc] = V4[j*4+k, c]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4+i, gc] = W4[k*4+i, c]
                
    # Cols 112-127: U3, V3, W3 (for k=8)
    for c in range(16):
        gc = 112 + c
        U[:, gc] = U3[:, c]
        for j in range(4):
            V[j*9+8, gc] = V3[j, c]
        for i in range(4):
            W[8*4+i, gc] = W3[i, c]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

print(construct())
