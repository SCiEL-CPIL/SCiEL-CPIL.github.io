"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen's algorithm recursively:
    # 4x4x5 matrix multiplication tensor.
    # Split m=4 into two 2x2 blocks: A is n x m (4x4), B is m x p (4x5).
    # Actually, let's use a standard approach: decompose using Kronecker products
    # or known rank decompositions.
    
    # The minimal rank for 4x4x4 is 49 (Strassen^2).
    # For 4x4x5, we can view it as two 4x4x4 multiplications plus some adjustments,
    # but a simpler approach: use the rank-49 decomposition for the first 4 columns of B,
    # and handle the 5th column separately? No, that's not how tensor rank works.
    
    # Actually, we can use a block decomposition. Let's use the standard result:
    # rank(4,4,5) <= rank(4,4,4) + rank(4,4,1) = 49 + 4*4*1 = 49 + 16 = 65?
    # That's not correct either. Tensor rank is subadditive but not in that simple way.
    
    # Let me use a different approach: use the explicit rank-49 decomposition for 4x4x4,
    # and extend it to 4x4x5 by treating the 5th column specially.
    # Actually, the cleanest way: use a general method.
    
    # For now, use the schoolbook approach but with Strassen for the 2x2 base case.
    # 4x4x5 = 80 via schoolbook.
    # With Strassen: 4 = 2+2, so we do 7 multiplications of (2x2) x (2x5) each?
    # No, Strassen works for square matrices. For rectangular, we can pad.
    
    # Let's use a direct rank decomposition.
    # For 4x4x5, a known upper bound is 65 using Strassen recursively.
    # 4x4x4 has rank 49. 4x4x5 can be seen as 4x4x4 + 4x4x1.
    # But tensor rank is not additive in that way.
    
    # Let me just use the rank-49 Strassen decomposition for the 4x4x4 part
    # and add the 5th column contributions.
    
    # Actually, the simplest correct approach: use Kronecker product of 2x2x2 Strassen.
    # The 4x4x4 tensor T_{i,j,k} where C[i,k] = sum_j A[i,j] B[j,k]
    # can be written as T = T2 ⊗ T2 where T2 is the 2x2x2 multiplication tensor.
    # Rank of T2 is 7, so rank of T4 is at most 7*7 = 49.
    
    # For 4x4x5: Let's decompose B into B1 (4x4) and B2 (4x1).
    # C = A*B1 + A*B2
    # A*B1 has rank <= 49. A*B2 (4x4 times 4x1 = 4x1 vector) has rank <= 4*4*1 = 16.
    # But these tensors live in different spaces... Actually, the output C is 4x5.
    # C[:, 0:4] = A*B1 (4x4), C[:, 4] = A*B2 (4x1).
    # So we can decompose C[:,0:4] with rank 49 and C[:,4] with rank 16.
    # Total rank <= 65.
    
    # Let me implement this.
    # First, the 4x4x4 part using Kronecker product of Strassen.
    
    def strassen_2x2():
        # Return U, V, W for 2x2x2 Strassen decomposition, rank 7.
        # A is 2x2: a00, a01, a10, a11 (index: i*m+j, m=2, so [0,1,2,3])
        # B is 2x2: b00, b01, b10, b11 (index: j*p+k, p=2, so [0,1,2,3])
        # C is 2x2: c00, c01, c10, c11 (index: k*n+i, n=2, so [0,1,2,3])
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
        
        U = [
            [1, 0, 0, 1],  # m0
            [0, 0, 1, 1],  # m1
            [1, 0, 0, 0],  # m2
            [0, 0, 0, 1],  # m3
            [1, 1, 0, 0],  # m4
            [1, 0, 1, 0],  # m5
            [0, 1, 0, 1],  # m6
        ]
        V = [
            [1, 0, 0, 1],  # m0
            [1, 0, 0, 0],  # m1
            [0, 1, 0, 1],  # m2
            [0, 0, 1, 1],  # m3
            [0, 0, 0, 1],  # m4
            [1, 1, 0, 0],  # m5
            [0, 0, 1, 1],  # m6
        ]
        W = [
            [1, 0, 0, 1],  # m0: c00 + c11
            [0, 0, 1, 0],  # m1: c10
            [0, 1, 0, 0],  # m2: c01
            [1, 0, 1, 0],  # m3: c00 + c10
            [1, 1, 0, 0],  # m4: c00 + c01  -- wait
            # Let me recheck:
            # c00 = m0 + m3 - m4 + m6
            # c01 = m2 + m4
            # c10 = m1 + m3
            # c11 = m0 - m1 + m2 + m5
            
            # m0: c00+=1, c11+=1 -> [1,0,0,1]
            # m1: c10+=1 -> [0,0,1,0]
            # m2: c01+=1, c11+=1 -> [0,1,0,1]
            # m3: c00+=1, c10+=1 -> [1,0,1,0]
            # m4: c00+=-1, c01+=1 -> [-1,1,0,0]
            # m5: c11+=1 -> [0,0,0,1]
            # m6: c00+=1 -> [1,0,0,0]
        ]
        W = [
            [1, 0, 0, 1],   # m0
            [0, 0, 1, 0],   # m1
            [0, 1, 0, 1],   # m2
            [1, 0, 1, 0],   # m3
            [-1, 1, 0, 0],  # m4
            [0, 0, 0, 1],   # m5
            [1, 0, 0, 0],   # m6
        ]
        return U, V, W
    
    U2, V2, W2 = strassen_2x2()
    
    # Now for 4x4x4, we use Kronecker product.
    # A is 4x4, indexed by (i,j) where i in 0..3, j in 0..3.
    # i = 2*i1 + i2, j = 2*j1 + j2.
    # B is 4x4, indexed by (j,k) where j in 0..3, k in 0..3.
    # j = 2*j1 + j2, k = 2*k1 + k2.
    # C is 4x4, indexed by (k,i) in trace convention: k*n+i = 4*k+i.
    # k = 2*k1 + k2, i = 2*i1 + i2.
    # C[k,i] = sum_{j} A[i,j] B[j,k]
    #         = sum_{j1,j2} A[2*i1+i2, 2*j1+j2] B[2*j1+j2, 2*k1+k2]
    # This decomposes into a Kronecker product of two 2x2x2 multiplications
    # when we group by (j1) as the "middle" index... actually it's more complex.
    # Standard result: the 4x4x4 multiplication tensor is T2 ⊗ T2.
    # The rank-49 decomposition comes from 7*7 terms.
    
    # For the Kronecker product structure:
    # Term (a,b) where a in 0..6, b in 0..6.
    # U[(i,j), (a,b)] = U2[i,j? ...]
    
    # Let me index A as a vector of length 16, index = i*4+j.
    # U2 has shape (4,7), indexed by [i2*2+j2? No, U2 is for 2x2, index is i*2+j = i2*2+j2.
    # For the 4x4 case, A index = i*4+j where i = 2*i1+i2, j = 2*j1+j2.
    # So A_index = (2*i1+i2)*4 + (2*j1+j2) = 8*i1 + 2*i2 + 2*j1 + j2.
    
    # The Kronecker product: U4[A_idx, (a,b)] = U2[ (i1,j1) part? 
    # Actually, for the tensor product of two bilinear forms, if T1 has rank decomposition
    # T1 = sum_r u1_r ⊗ v1_r ⊗ w1_r, then T1 ⊗ T2 = sum_{r,s} (u1_r ⊗ u2_s) ⊗ (v1_r ⊗ v2_s) ⊗ (w1_r ⊗ w2_s).
    
    # For matrix multiplication, T_{i,j,k} = delta(i,j) * delta(j,k)... no.
    # The multiplication tensor T has T[i,j,k] = 1 if j=j (always), so it's just the "matrix unit" tensor.
    # For 2x2x2: T[a00,a01,a10,a11; b00,b01,b10,b11; c00,c01,c10,c11] with C=A*B.
    # This is a specific tensor, not a Kronecker product of smaller multiplication tensors in an obvious way.
    
    # Let me just hardcode the rank-49 decomposition using the known result.
    # Or better: use a different approach. Let's use the rank-65 decomposition.
    
    # Schoolbook rank for 4x4x5 is 80. Let me just return 80 for now to ensure correctness,
    # but that's the same as the seed. I need something better.
    
    # Let me implement the Strassen-based rank-49 for 4x4x4 and rank-16 for the 5th column.
    # Total: 65.
    
    # 4x4x4 part: C[:,0:4] = A * B[:,0:4]
    # 4x4x1 part: C[:,4] = A * B[:,4]
    
    # For the 4x4x4 part, use Strassen recursively (rank 49).
    # For the 4x4x1 part, use schoolbook (rank 16).
    
    rank_total = 65
    U_out = [[0] * rank_total for _ in range(16)]  # n*m = 16
    V_out = [[0] * rank_total for _ in range(20)]  # m*p = 20
    W_out = [[0] * rank_total for _ in range(20)]  # p*n = 20
    
    # First 49 terms: 4x4x4 Strassen
    # Let me define the Strassen decomposition for 4x4x4 explicitly.
    
    # A index: i*4+j, i in 0..3, j in 0..3
    # B index: j*5+k, j in 0..3, k in 0..4 (for the full 4x5 case)
    # For 4x4x4: B index: j*4+k, j in 0..3, k in 0..3
    # C index: k*4+i, k in 0..3, i in 0..3
    
    # Strassen on 4x4x4: split into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # A[i,j] = A_block[i//2, j//2][(i%2), (j%2)]
    
    # Let me just enumerate all 49 terms explicitly.
    # Term (s, t) where s, t in 0..6.
    # U[(i,j), (s,t)] = U2[i1,j1? ...]
    
    # For the Kronecker product interpretation:
    # A vector: index = (i1,i2,j1,j2) flattened as i1*8 + i2*4 + j1*2 + j2? No.
    # A[i,j] with i=2*i1+i2, j=2*j1+j2.
    # Index in vector: i*4+j = (2*i1+i2)*4 + (2*j1+j2) = 8*i1 + 4*i2 + 2*j1 + j2.
    
    # U2[u2_idx, s] where u2_idx = i2*2+j2.
    # For term (s,t): U4[u4_idx, (s,t)] = U2[u2_hi, s] * U2[u2_lo, t] where
    # u4_idx = 8*i1 + 4*i2 + 2*j1 + j2
    # u2_hi corresponds to (i1, j1) and u2_lo corresponds to (i2, j2)?
    # Not exactly. The Kronecker product means:
    # U4 = U2 ⊗ U2 (in some sense).
    
    # Let me define it as: for each (i1, i2, j1, j2) and (s, t):
    # U4[i1*8 + i2*4 + j1*2 + j2, s*7+t] = U2[i1*2+j1, s] * U2[i2*2+j2, t]
    
    # Wait, that's not right either. In the Kronecker product of tensors,
    # if T = T1 ⊗ T2, then T[i1,i2;j1,j2;k1,k2] = T1[i1;j1;k1] * T2[i2;j2;k2].
    # For matrix multiplication, C[i,k] = sum_j A[i,j] B[j,k].
    # If we write i = 2*i1+i2, j = 2*j1+j2, k = 2*k1+k2:
    # C[2*i1+i2, 2*k1+k2] = sum_{j1,j2} A[2*i1+i2, 2*j1+j2] B[2*j1+j2, 2*k1+k2]
    # This is NOT a Kronecker product of two 2x2x2 multiplication tensors because
    # the sum is over both j1 and j2, mixing the indices.
    
    # So the simple Kronecker product doesn't apply directly. The rank-49 result
    # for 4x4x4 comes from a different argument (Strassen recursively applied).
    
    # Let me just use a known rank-49 decomposition. Since I can't easily derive it,
    # let me use a different strategy: use Strassen directly on the 4x4x4 multiplication.
    
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # B = [[B11, B12], [B21, B22]] where each is 2x2.
    # C = A*B:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Using Strassen:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Each Px is a product of two 2x2 matrices, so each has rank <= 7.
    # Total rank <= 7*7 = 49.
    
    # For the 5th column of B, we just do schoolbook: rank 4*4 = 16.
    
    # Let me code this up.
    col = 0
    for s in range(7):
        for t in range(7):
            # This term corresponds to the (s,t)-th rank-1 tensor in the
            # Strassen decomposition of the 4x4x4 multiplication.
            # U component: for each (i,j) in 4x4, the coefficient.
            # We need to express the coefficient of A[i,j] in the s-th Strassen
            # product for the "outer" 2x2, times the coefficient for the "inner" 2x2.
            
            # This is getting complex. Let me just use the schoolbook rank-80
            # to ensure correctness, since I can't quickly derive the exact
            # rank-49 decomposition.
            
            pass
    
    # Given time constraints, let me return the schoolbook decomposition.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
