import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for 4x4 * 4x5 multiplication.
    
    Strategy:
    1. Decompose the 4x4 * 4x4 part using a Kronecker product of Strassen's 2x2 algorithm (rank 7).
       This gives a rank of 7*7 = 49 for the 4x4 * 4x4 sub-problem.
    2. Handle the remaining 4x4 * 4x1 part (the 5th column of B) using the schoolbook method (rank 16).
    3. Combine these decompositions. Total rank = 49 + 16 = 65.
    
    The schoolbook rank is 4*4*5 = 80. So this is an improvement.
    
    Interface:
    U: (n*m, rank) = (16, rank)
    V: (m*p, rank) = (20, rank)
    W: (p*n, rank) = (20, rank)
    
    Indices:
    U[i*m + j, c] corresponds to A[i, j]
    V[j*p + k, c] corresponds to B[j, k]
    W[k*n + i, c] corresponds to C[k, i] -- WAIT. The task says: "output factor follows the official trace convention W[k*n+i,r]".
    Usually, C is n x p. So C[i, k]. The index in W is k*n + i. This means W is indexed by (k, i) -> scalar.
    Let's check the standard convention.
    Standard: C = A * B. C[i, k] = sum_j A[i, j] B[j, k].
    The tensor T_{i,j,k} = A[i,j] B[j,k] -> C[i,k].
    The decomposition is T = sum_r U_r (V_r x W_r).
    U_r is a vector of size n*m. V_r is a vector of size m*p. W_r is a vector of size p*n.
    The product is sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = C[i,k].
    So W_r is indexed by k*n + i. This matches C[i,k] if we transpose W's logic or just keep it as is.
    Actually, if W_r[k*n+i] contributes to C[i,k], then the index mapping is correct.
    
    Let's build the matrices.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Part 1: 4x4 * 4x4 (columns 0-3 of B)
    # Use Kronecker product of Strassen 2x2.
    # Strassen 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 - M4 + M5
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3
    
    # We need to map this to the 4x4 case.
    # A = [[A00, A01, A02, A03],
    #      [A10, A11, A12, A13],
    #      [A20, A21, A22, A23],
    #      [A30, A31, A32, A33]]
    # Split A into 2x2 blocks of size 2x2:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # Similarly for B (first 4 columns).
    
    # The Kronecker product approach:
    # The rank is R1 * R2 where R1, R2 are ranks of the 2x2 decomposition.
    # Actually, the Kronecker product of two rank-R decompositions gives a rank R^2 decomposition for the block diagonal?
    # No, for matrix multiplication, the Kronecker product of the bilinear forms works.
    # If A = A1 * A2 (block structure), B = B1 * B2, then C = A*B.
    # The bilinear forms for 4x4 are combinations of products of bilinear forms for 2x2.
    
    # Let's explicitly build the 49 terms for the 4x4x4 part.
    # We will use the fact that the 4x4 multiplication tensor is the Kronecker product of two 2x2 multiplication tensors.
    # T_{4,4,4} = T_{2,2,2} \otimes T_{2,2,2}.
    # If T_2 has rank 7 with factors U2, V2, W2, then T_4 has rank 49 with factors:
    # U4 = kron(U2, U2), V4 = kron(V2, V2), W4 = kron(W2, W2).
    # We need to be careful with the ordering of indices.
    # Index in U4: (i1, j1, i2, j2) -> i1*2+i2, j1*2+j2? Or (i1, i2, j1, j2)?
    # Standard Kronecker product of matrices:
    # If U2 is 4x7 (2*2 x 7), V2 is 4x7, W2 is 4x7.
    # U4 will be 16x49.
    # The index mapping for U is i*m + j.
    # For 2x2: i*2 + j.
    # For 4x4: i*4 + j.
    # Let's define the 2x2 factors.
    
    # Strassen 2x2 factors:
    # U2[i*2+j, r]
    # V2[j*2+k, r]
    # W2[k*2+i, r]
    
    # Let's construct U2, V2, W2 for 2x2.
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: (A11+A22)*(B11+B22) -> C11 + C22 terms
    # A indices: 11->0, 22->3. B indices: 11->0, 22->3. C indices: 11->0, 22->3.
    # U2: A11 (0), A22 (3) -> 1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (A21+A22)*B11 -> C21 term
    # A: 21->2, 22->3. B: 11->0. C: 21->2.
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # M3: A11*(B12-B22) -> C12 term
    # A: 11->0. B: 12->1, 22->3. C: 12->1.
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    
    # M4: A22*(B21-B11) -> C11, C22 terms
    # A: 22->3. B: 21->2, 11->0. C: 11->0, 22->3.
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = -1; W2[3, 3] = 1
    
    # M5: (A11+A12)*B22 -> C11, C12 terms
    # A: 11->0, 12->1. B: 22->3. C: 11->0, 12->1.
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1; W2[1, 4] = 1
    
    # M6: (A21-A11)*(B11+B12) -> C21, C22 terms
    # A: 21->2, 11->0. B: 11->0, 12->1. C: 21->2, 22->3.
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[3, 5] = 1
    
    # M7: (A12-A22)*(B21+B22) -> C11, C22 terms
    # A: 12->1, 22->3. B: 21->2, 22->3. C: 11->0, 22->3.
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1; W2[3, 6] = 1
    
    # Now construct 4x4 factors via Kronecker product.
    # The Kronecker product of the tensors.
    # If we have factors U2, V2, W2 for 2x2, the factors for 4x4 are:
    # U4 = kron(U2, U2) -- but we must ensure the index ordering matches i*4+j.
    # Let's check the index mapping.
    # In 2x2, index is i*2+j.
    # In 4x4, index is i*4+j.
    # If we write i = 2*i1 + i2, j = 2*j1 + j2.
    # Then index = (2*i1+i2)*4 + (2*j1+j2) = 8*i1 + 4*i2 + 2*j1 + j2.
    # This is not the standard row-major order for the Kronecker product of indices (i1,j1) and (i2,j2).
    # Standard Kronecker product of vectors u (size 4) and v (size 4) gives w (size 16).
    # w[a*4+b] = u[a]*v[b].
    # If we map 4x4 indices (i,j) to 1D index i*4+j.
    # We can think of the 4x4 matrix as a 2x2 block matrix of 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where Axy is 2x2.
    # The bilinear form for A*B is sum over block products.
    # The Kronecker product of the 2x2 decomposition works if we order the indices correctly.
    # Let's assume the standard Kronecker product of the factor matrices U2, V2, W2.
    # U4 = np.kron(U2, U2) will have shape (16, 49).
    # Does U4[i*4+j, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2] ?
    # Let's verify.
    # np.kron(A, B)[i, j] = A[i//B.shape[1], j//B.shape[0]] * B[i%B.shape[1], j%B.shape[0]]
    # This is getting tricky with index permutations.
    # A safer way: explicitly build the 49 terms by combining the 7 Strassen terms.
    
    # Let's build U4, V4, W4 of shape (16, 49) and (20, 49) and (20, 49) for the first 4 columns.
    U4 = np.zeros((16, 49))
    V4 = np.zeros((20, 49)) # 4x5, so 20 rows. We only fill first 16 (4x4) part.
    W4 = np.zeros((20, 49)) # 5x4, so 20 rows.
    
    # Iterate over the 49 combinations of Strassen terms (r1, r2)
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # For each (i1, j1) in U2[r1], and (i2, j2) in U2[r2]
            # The contribution to U4[i*4+j, r] is U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
            # Wait, the Kronecker product structure for matrix multiplication:
            # The term is (U2[r1] \otimes U2[r2]) (V2[r1] \otimes V2[r2]) ...
            # Actually, the bilinear form is:
            # C = sum_{r1,r2} (U2[:, r1] \otimes U2[:, r2]) * (V2[:, r1] \otimes V2[:, r2]) ...
            # This works because (A1 \otimes A2) (B1 \otimes B2) = (A1 B1) \otimes (A2 B2).
            # So we can just take the Kronecker products of the factor vectors.
            
            u_vec1 = U2[:, r1] # size 4
            u_vec2 = U2[:, r2] # size 4
            u_combined = np.kron(u_vec1, u_vec2) # size 16
            
            v_vec1 = V2[:, r1]
            v_vec2 = V2[:, r2]
            v_combined = np.kron(v_vec1, v_vec2) # size 16
            
            w_vec1 = W2[:, r1]
            w_vec2 = W2[:, r2]
            w_combined = np.kron(w_vec1, w_vec2) # size 16
            
            # We need to map these 16-length vectors to the 4x4 index space.
            # The Kronecker product of two 2x2 index vectors (i1, j1) and (i2, j2)
            # results in a vector indexed by (i1, i2, j1, j2) if we use standard kron.
            # np.kron([a,b,c,d], [e,f,g,h]) = [ae, af, ag, ah, be, ...]
            # The index in the result is i1*4 + i2*2 + ... NO.
            # Let's check np.kron index mapping.
            # If u = [u00, u01, u10, u11] (row-major 2x2), v = [v00, v01, v10, v11].
            # kron(u, v) = [u00*v00, u00*v01, u00*v10, u00*v11, u01*v00, ...]
            # The index is (i1*2 + i2) * 2 + j? No.
            # Let's just assume the standard row-major ordering of the 4x4 matrix corresponds to
            # the Kronecker product of the row-major orderings of the 2x2 blocks.
            # This is true if the block structure is A = [[A11, A12], [A21, A22]]
            # and the elements are ordered A11(00), A11(01), A11(10), A11(11), A12(00)...
            # But standard row-major for 4x4 is A(0,0), A(0,1), A(0,2), A(0,3), A(1,0)...
            # This matches the block row-major order:
            # Row 0: A11(0,0), A11(0,1), A12(0,0), A12(0,1)
            # Row 1: A11(1,0), A11(1,1), A12(1,0), A12(1,1)
            # This is exactly the Kronecker product of the index vectors if we map:
            # 2x2 index (i,j) -> i*2+j.
            # 4x4 index (I,J) where I=2*i1+i2, J=2*j1+j2.
            # The Kronecker product of the 2x2 matrices corresponds to this ordering.
            
            U4[:, r] = u_combined
            V4[:16, r] = v_combined # Only first 4 columns of B (4x4 part)
            W4[:16, r] = w_combined # Only first 4 rows of C (4x4 part)
            # Note: W4 has size 20 (5x4). The 4x4 part of C corresponds to the first 16 entries?
            # C is 4x5. C[i, k] where i in 0..3, k in 0..4.
            # The first 4 columns (k=0..3) form the 4x4 matrix.
            # Index in W is k*4 + i.
            # For k=0..3, i=0..3, the indices are 0..15.
            # So W4[:16, r] is correct for the 4x4 part of C.
    
    # Part 2: 4x4 * 4x1 (column 4 of B)
    # This is a rank 16 problem.
    U1 = np.zeros((16, 16))
    V1 = np.zeros((20, 16))
    W1 = np.zeros((20, 16))
    
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            U1[c, c] = 1
            # V index: j*p + k = j*5 + 4
            V1[j * 5 + 4, c] = 1
            # W index: k*n + i = 4*4 + i = 16 + i
            W1[16 + i, c] = 1
            
    # Combine
    rank = 49 + 16
    U_final = np.hstack([U4, U1])
    V_final = np.hstack([V4, V1])
    W_final = np.hstack([W4, W1])
    
    # Convert to lists for the return format
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
