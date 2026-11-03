"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-49 decomposition using Kronecker product of Strassen's rank-7 decomposition for 2x2.
    # This is an improvement over the schoolbook rank-64.
    n, m, p = DIMENSIONS
    
    # Strassen's 7 rank-1 terms for 2x2 matrix multiplication.
    # Indices: u[i*2+j], v[j*2+k], w[k*2+i] (following the W[k*n+i] convention from seed, 
    # but for 2x2, W index is k*2+i).
    # Note: The seed uses W[k*n+i]. For 2x2, this is W[k*2+i].
    # Standard Strassen terms for C = AB:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We define the vectors u, v, w for each term.
    # u is length 4 (A11, A12, A21, A22)
    # v is length 4 (B11, B12, B21, B22)
    # w is length 4 (C11, C12, C21, C22) -> BUT seed uses W[k*2+i].
    # Seed W index: k*2+i.
    # C11 is i=0, k=0 -> idx 0.
    # C12 is i=0, k=1 -> idx 1. (Wait, C12 is row 1 col 2? No, C12 is row 1 col 2 in 1-based. 
    # In 0-based: C[0,1]. i=0, k=1. W index = 1*2 + 0 = 2.
    # C21 is row 2 col 1. i=1, k=0. W index = 0*2 + 1 = 1.
    # C22 is row 2 col 2. i=1, k=1. W index = 1*2 + 1 = 3.
    # So W vector order: [C11, C21, C12, C22] corresponding to indices [0, 1, 2, 3].
    
    # Let's verify the seed convention: W[k*n+i].
    # For n=2:
    # k=0, i=0 -> 0 (C00)
    # k=0, i=1 -> 1 (C10) -> This is C[1,0] in standard math?
    # Standard C[i,k].
    # If W index is k*n+i, then index 1 is k=0, i=1. This corresponds to C[1,0] (row 1, col 0).
    # Standard multiplication: C[i,k] = sum A[i,j] B[j,k].
    # So C[1,0] is row 1, col 0.
    # So the W vector is [C00, C10, C01, C11].
    
    # Strassen terms for C00, C10, C01, C11:
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # We need to express each Mr as u_r, v_r, w_r.
    # The decomposition is sum of rank-1 tensors.
    # Each Mr corresponds to one rank-1 term? No, Strassen uses 7 multiplications, 
    # but the output is a linear combination.
    # The rank-1 decomposition of the tensor is the sum of 7 terms.
    # Each term is u_r * v_r * w_r.
    # For Strassen, the 7 terms are:
    # 1. u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) ? No.
    # The tensor decomposition is:
    # T = sum_{r=1}^7 u_r \otimes v_r \otimes w_r.
    
    # The 7 vectors for 2x2 (using indices 0..3 for A, B, C):
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 10, 01, 11 (based on W[k*2+i])
    
    # Term 1: M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # This is not a rank-1 term in the tensor sense unless we split it.
    # Actually, the rank-1 decomposition of M2 is known.
    # I will use the standard 7 vectors.
    
    # u1 = [1, 0, 0, 1]
    # v1 = [1, 0, 0, 1]
    # w1 = [1, 0, 0, 1] (C00, C11) -> Wait, w1 has non-zeros at C00 and C11.
    # This is not a rank-1 vector w1. w1 is a vector.
    # In the tensor sum, w_r is a vector.
    # For term 1:
    # u1 = [1, 0, 0, 1]
    # v1 = [1, 0, 0, 1]
    # w1 = [1, 0, 0, 1] -> This means C00 += 1, C11 += 1.
    # Does M1 contribute to C00 and C11?
    # C00 = M1 + ...
    # C11 = M1 - ...
    # Yes.
    
    # Let's list the 7 (u, v, w) for 2x2.
    # 1. u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # 2. u=[0,0,1,1], v=[1,0,0,0], w=[0,1,0,0] (M2: (A10+A11)B00 -> C10)
    #    w index for C10 is 1. So w=[0,1,0,0].
    # 3. u=[1,0,0,0], v=[0,1,0,-1], w=[0,0,1,0] (M3: A00(B01-B11) -> C01)
    #    w index for C01 is 2. So w=[0,0,1,0].
    # 4. u=[0,0,0,1], v=[0,0,1,-1], w=[1,1,0,0] (M4: A11(B10-B00) -> C00, C10)
    #    M4 contributes to C00 (idx 0) and C10 (idx 1).
    #    w=[1,1,0,0].
    # 5. u=[1,1,0,0], v=[0,0,0,1], w=[-1,0,1,0] (M5: (A00+A01)B11 -> C00, C01)
    #    M5 contributes to C00 (idx 0) and C01 (idx 2).
    #    w=[-1,0,1,0].
    # 6. u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1] (M6: (A10-A00)(B00+B01) -> C11)
    #    M6 contributes to C11 (idx 3).
    #    w=[0,0,0,1].
    # 7. u=[0,-1,0,1], v=[0,0,1,1], w=[1,0,0,1] (M7: (A01-A11)(B10+B11) -> C00, C11)
    #    M7 contributes to C00 (idx 0) and C11 (idx 3).
    #    w=[1,0,0,1].
    
    # Let's verify C00:
    # T1: 1
    # T4: 1
    # T5: -1
    # T7: 1
    # Sum: 1+1-1+1 = 2? No, C00 = M1 + M4 - M5 + M7.
    # Coefficients:
    # M1: 1
    # M4: 1
    # M5: -1
    # M7: 1
    # Sum: 1+1-1+1 = 2?
    # Wait, Strassen: C00 = M1 + M4 - M5 + M7.
    # If I sum the w vectors:
    # w1[0]=1, w4[0]=1, w5[0]=-1, w7[0]=1. Sum = 2.
    # But C00 should be 1 * A00B00 + ...
    # The tensor entry for A00B00->C00 is 1.
    # The sum of products u*v*w for a specific A00, B00, C00 should be 1.
    # If I set A00=1, B00=1, others 0.
    # T1: u1[0]=1, v1[0]=1, w1[0]=1 -> 1
    # T4: u4[0]=0 -> 0
    # T5: u5[0]=1, v5[0]=0 -> 0
    # T7: u7[0]=0 -> 0
    # Sum = 1. Correct.
    
    # So the 7 vectors are correct for 2x2.
    
    # Now, construct 4x4 by Kronecker product.
    # U_4 = kron(U_2, U_2) ?
    # The indices for 4x4 are i,j in 0..3.
    # i = 2*i1 + i0, j = 2*j1 + j0.
    # u_4[i, j] = u_2[i1, j1] * u_2[i0, j0].
    # Similarly for v and w.
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    u2 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, -1, 0, 1]
    ]
    v2 = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w2 = [
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1]
    ]
    
    col = 0
    for r1 in range(7):
        for r2 in range(7):
            # u_4 index: i*4+j = (2*i1+i0)*4 + (2*j1+j0) = 8*i1 + 2*i0 + 4*j1 + 2*j0?
            # No. i = 2*i1 + i0. j = 2*j1 + j0.
            # idx = i*4 + j = (2*i1+i0)*4 + 2*j1 + j0 = 8*i1 + 4*i0 + 2*j1 + j0.
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            idx_u = (2*i1 + i0)*4 + (2*j1 + j0)
                            val_u = u2[r1][i1*2+j1] * u2[r2][i0*2+j0]
                            U[idx_u][col] += val_u
                            
                            idx_v = (2*j1 + j0)*4 + (2*k1 + k0) # Wait, v depends on j, k.
                            # v index in 2x2 is j*2+k.
                            # In 4x4, v index is j*4+k.
                            # j = 2*j1+j0, k = 2*k1+k0.
                            # idx_v = j*4+k = (2*j1+j0)*4 + 2*k1 + k0 = 8*j1 + 4*j0 + 2*k1 + k0.
                            
                            # We need to sum over k1, k0 for V.
                            # But V is a matrix. We fill V[idx_v][col].
                            # So we iterate k1, k0.
            # This nested loop is inefficient. Let's precompute the vectors.
            
    # Precompute u4, v4, w4 vectors for each (r1, r2).
    u4_vecs = []
    v4_vecs = []
    w4_vecs = []
    
    for r1 in range(7):
        for r2 in range(7):
            u4 = [0]*16
            v4 = [0]*16
            w4 = [0]*16
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            idx_u = (2*i1 + i0)*4 + (2*j1 + j0)
                            u4[idx_u] = u2[r1][i1*2+j1] * u2[r2][i0*2+j0]
                            
                            idx_v = (2*j1 + j0)*4 + (2*k1 + k0) # k is free here? No.
                            # v4 depends on j, k.
                            # We need to fill v4 for all j, k.
                            # So iterate k1, k0.
            # This is getting messy. I will just use the Kronecker product logic directly in the loops.
            
    # Simpler:
    for r1 in range(7):
        for r2 in range(7):
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            for k1 in range(2):
                                for k0 in range(2):
                                    # U
                                    idx_u = (2*i1 + i0)*4 + (2*j1 + j0)
                                    U[idx_u][col] += u2[r1][i1*2+j1] * u2[r2][i0*2+j0]
                                    
                                    # V
                                    idx_v = (2*j1 + j0)*4 + (2*k1 + k0)
                                    V[idx_v][col] += v2[r1][j1*2+k1] * v2[r2][j0*2+k0]
                                    
                                    # W
                                    # W index in 2x2 is k*2+i.
                                    # In 4x4, W index is k*4+i.
                                    # k = 2*k1+k0, i = 2*i1+i0.
                                    idx_w = (2*k1 + k0)*4 + (2*i1 + i0)
                                    W[idx_w][col] += w2[r1][k1*2+i1] * w2[r2][k0*2+i0]
                                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
