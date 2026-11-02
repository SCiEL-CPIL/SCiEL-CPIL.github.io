import numpy as np

def construct():
    n, m, p = 5, 5, 5
    
    # Step 1: 2x2 Strassen Decomposition
    # U, V, W shapes (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    U2[0, 0] += 1; U2[3, 0] += 1
    V2[0, 0] += 1; V2[3, 0] += 1
    W2[0, 0] += 1; W2[3, 0] += 1
    
    # M2 = (A10+A11)B00 -> C10, C11
    U2[2, 1] += 1; U2[3, 1] += 1
    V2[0, 1] += 1
    W2[2, 1] += 1; W2[3, 1] += 1
    
    # M3 = A00(B01-B11) -> C01
    U2[0, 2] += 1
    V2[1, 2] += 1; V2[3, 2] -= 1
    W2[1, 2] += 1
    
    # M4 = A11(B10-B00) -> C10, C11
    U2[3, 3] += 1
    V2[2, 3] += 1; V2[0, 3] -= 1
    W2[2, 3] += 1; W2[3, 3] += 1
    
    # M5 = (A00+A01)B11 -> C00, C01
    U2[0, 4] += 1; U2[1, 4] += 1
    V2[3, 4] += 1
    W2[0, 4] -= 1; W2[1, 4] += 1
    
    # M6 = (A10-A00)(B00+B01) -> C11
    U2[2, 5] += 1; U2[0, 5] -= 1
    V2[0, 5] += 1; V2[1, 5] += 1
    W2[3, 5] += 1
    
    # M7 = (A01-A11)(B10+B11) -> C00
    U2[1, 6] += 1; U2[3, 6] -= 1
    V2[2, 6] += 1; V2[3, 6] += 1
    W2[0, 6] += 1

    # Step 2: Tensor Product to 4x4
    # Rank 49
    # U4 shape (16, 49), V4 shape (16, 49), W4 shape (16, 49)
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # U4[i*4+j, r] = U2[i_block*2+i_inner, r1] * U2[j_block*2+j_inner, r2]
            # i = 2*i_block + i_inner
            # j = 2*j_block + j_inner
            for i_block in range(2):
                for j_block in range(2):
                    for i_inner in range(2):
                        for j_inner in range(2):
                            i = i_block * 2 + i_inner
                            j = j_block * 2 + j_inner
                            val = U2[i_block * 2 + i_inner, r1] * U2[j_block * 2 + j_inner, r2]
                            # Wait, the tensor product structure is:
                            # A is 4x4. Viewed as 2x2 blocks of 2x2.
                            # A[i,j] where i=2*i1+i2, j=2*j1+j2.
                            # The term in sum is A[i,j] B[j,k] C[i,k].
                            # This is sum over j1, j2.
                            # The decomposition is sum over r1, r2.
                            # Term (r1, r2) contributes:
                            # u1 (for A block i1, j1) and u2 (for A inner i2, j2)?
                            # No, the tensor product of decompositions:
                            # T4 = T2 (i1,j1,k1) x T2 (i2,j2,k2)
                            # A[i1,i2, j1,j2] B[j1,j2, k1,k2] C[i1,i2, k1,k2]
                            # Sum over j1,j2.
                            # Decomposition:
                            # Sum_{r1,r2} (U2[i1*2+i2, r1] * U2[j1*2+j2, r2]) ... ?
                            # No.
                            # The Kronecker product of the tensors.
                            # If T2 = sum_r u_r v_r w_r
                            # Then T4 = sum_{r1,r2} (u_{r1} \otimes u_{r2}) (v_{r1} \otimes v_{r2}) (w_{r1} \otimes w_{r2})
                            # Here u_r is a vector of length 4 (for A).
                            # u_{r1} \otimes u_{r2} is a vector of length 16.
                            # Index of A is (i1,i2) mapped to i1*2+i2?
                            # No, A is 4x4. Index is i, j.
                            # i = 2*i1 + i2.
                            # The vector u_r has components for i, j?
                            # In the 2x2 case, U2 is (4, 7). Rows are i*2+j.
                            # So u_r is a vector of length 4 (flattened 2x2 matrix).
                            # The Kronecker product of two 2x2 matrices is a 4x4 matrix.
                            # The flattened Kronecker product corresponds to the indices.
                            # If A = A1 \otimes A2 (conceptually), then A[i,j] = A1[i1,j1] A2[i2,j2].
                            # But A is arbitrary.
                            # The tensor T4 is the "multiplication" of 4x4 matrices.
                            # This is isomorphic to T2 \otimes T2.
                            # So the decomposition is indeed the Kronecker product of the decompositions.
                            
                            # Let's check the index mapping.
                            # U4 row index: i*4+j.
                            # i = 2*i1 + i2, j = 2*j1 + j2.
                            # U4[i*4+j, r1*7+r2] = U2[i1*2+i2, r1] * U2[j1*2+j2, r2]?
                            # No, U2 row is i*2+j.
                            # The "u" vector in the decomposition is the A part.
                            # For 2x2, u_r is a 4-vector (A flattened).
                            # For 4x4, u_{r1,r2} is a 16-vector (A flattened).
                            # The 16-vector is the Kronecker product of the two 4-vectors?
                            # If A = A1 \otimes A2, then A[i,j] = A1[i1,j1] A2[i2,j2].
                            # But we are decomposing the *operation*, not a specific A.
                            # The tensor T is fixed.
                            # T4[i,j,k] = delta_{i1,k1} delta_{i2,k2} ... ?
                            # T4[i,j,k] = 1 if i=i1*2+i2, j=j1*2+j2, k=k1*2+k2 and j1=j1', j2=j2'?
                            # No, T4[i,j,k] is 1 if the indices match for the multiplication.
                            # T4 is the identity tensor for the bilinear map.
                            # T4[i,j,k] = 1 if i, j, k are valid indices for A[i,j] B[j,k] C[i,k].
                            # Actually, T4 is a 4x4x4x4x4x4 tensor? No, 4x4x4.
                            # T4[i,j,k] is not a scalar. It's a tensor of order 3 with dimensions 16,16,16.
                            # T4[A_idx, B_idx, C_idx].
                            # A_idx = i*4+j. B_idx = j*4+k. C_idx = k*4+i.
                            # T4 is 1 if the indices are consistent.
                            # Consistency: A's col j must match B's row j.
                            # A's row i must match C's row i.
                            # B's col k must match C's col k.
                            
                            # In the Kronecker product view:
                            # A_idx = (i1,i2) and (j1,j2).
                            # B_idx = (j1,j2) and (k1,k2).
                            # C_idx = (k1,k2) and (i1,i2).
                            # The tensor is T2(i1,j1,k1) * T2(i2,j2,k2).
                            # So U4[A_idx, r1,r2] = U2[A_idx_1, r1] * U2[A_idx_2, r2].
                            # A_idx_1 is the index in the first 2x2 block.
                            # A is 4x4.
                            # A_idx = i*4+j.
                            # i = 2*i1 + i2. j = 2*j1 + j2.
                            # A_idx_1 = i1*2 + j1.
                            # A_idx_2 = i2*2 + j2.
                            
                            # So:
                            val = U2[i1*2 + j1, r1] * U2[i2*2 + j2, r2]
                            U4[i*4 + j, r] = val
                            
            # Similarly for V and W
            for j_block in range(2):
                for k_block in range(2):
                    for j_inner in range(2):
                        for k_inner in range(2):
                            j = j_block * 2 + j_inner
                            k = k_block * 2 + k_inner
                            j1 = j_block; j2 = j_inner
                            k1 = k_block; k2 = k_inner
                            val = V2[j1*2 + k1, r1] * V2[j2*2 + k2, r2]
                            V4[j*4 + k, r] = val
                            
            for k_block in range(2):
                for i_block in range(2):
                    for k_inner in range(2):
                        for i_inner in range(2):
                            k = k_block * 2 + k_inner
                            i = i_block * 2 + i_inner
                            k1 = k_block; k2 = k_inner
                            i1 = i_block; i2 = i_inner
                            val = W2[k1*2 + i1, r1] * W2[k2*2 + i2, r2]
                            W4[k*4 + i, r] = val

    # Step 3: Embed into 5x5
    # Rank 94
    U5 = np.zeros((25, 94), dtype=int)
    V5 = np.zeros((25, 94), dtype=int)
    W5 = np.zeros((25, 94), dtype=int)
    
    # Copy 4x4 part
    # Indices 0..3 for i, j, k.
    # U5[i*5+j, r] = U4[i*4+j, r]
    # V5[j*5+k, r] = V4[j*4+k, r]
    # W5[k*5+i, r] = W4[k*4+i, r]
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U5[i*5+j, r] = U4[i*4+j, r]
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V5[j*5+k, r] = V4[j*4+k, r]
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W5[k*5+i, r] = W4[k*4+i, r]
                
    # Add cross terms
    # Top-right: C[i,4] = sum_j A[i,j]B[j,4] + A[i,4]B[4,4]
    # 16 terms for A[i,j]B[j,4]
    r_idx = 49
    for i in range(4):
        for j in range(4):
            U5[i*5+j, r_idx] = 1
            V5[j*5+4, r_idx] = 1
            W5[4*5+i, r_idx] = 1
            r_idx += 1
            
    # 4 terms for A[i,4]B[4,4]
    for i in range(4):
        U5[i*5+4, r_idx] = 1
        V5[4*5+4, r_idx] = 1
        W5[4*5+i, r_idx] = 1
        r_idx += 1
        
    # Bottom-left: C[4,j] = sum_i A[4,i]B[i,j] + A[4,4]B[4,j]
    # 16 terms for A[4,i]B[i,j]
    for i in range(4):
        for j in range(4):
            U5[4*5+i, r_idx] = 1
            V5[i*5+j, r_idx] = 1
            W5[j*5+4, r_idx] = 1
            r_idx += 1
            
    # 4 terms for A[4,4]B[4,j]
    for j in range(4):
        U5[4*5+4, r_idx] = 1
        V5[4*5+j, r_idx] = 1
        W5[j*5+4, r_idx] = 1
        r_idx += 1
        
    # Bottom-right: C[4,4] = sum_i A[4,i]B[i,4] + A[4,4]B[4,4]
    # 4 terms for A[4,i]B[i,4]
    for i in range(4):
        U5[4*5+i, r_idx] = 1
        V5[i*5+4, r_idx] = 1
        W5[4*5+4, r_idx] = 1
        r_idx += 1
        
    # 1 term for A[4,4]B[4,4]
    U5[4*5+4, r_idx] = 1
    V5[4*5+4, r_idx] = 1
    W5[4*5+4, r_idx] = 1
    r_idx += 1
    
    # r_idx should be 94
    assert r_idx == 94, f"Rank mismatch: {r_idx}"
    
    return {"U": U5.tolist(), "V": V5.tolist(), "W": W5.tolist()}
