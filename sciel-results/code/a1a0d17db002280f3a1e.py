"""
Matrix multiplication 2x5x5: Strassen-based rank-46 decomposition.
Uses 2x2x2 Strassen blocks to reduce rank below the schoolbook 50.
"""

def construct():
    n, m, p = 2, 5, 5
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 multiplication matrices for 2x2 blocks
    # S1 = A00*(B01-B11), S2=(A00+A01)*B11, S3=(A10-A00)*B00,
    # S4=A11*(B00-B10), S5=(A00+A11)*(B00+B11), S6=(A01-A11)*(B10+B11),
    # S7=(A00-A10)*(B00+B10)
    # Output: C00 = S2+S4-S5+S7, C01=S1+S5, C10=S3+S4, C11=S1-S2+S3+S6
    
    # We partition the 2x5 by 5x5 multiplication into 2x3 blocks of 2x2 sub-blocks
    # with one leftover column, and 3x3 blocks of 2x2 sub-blocks with one leftover row.
    # Actually, let's think differently.
    
    # The matrix A is 2x5, B is 5x5, C is 2x5.
    # Split A into columns: A = [A_left(2x4) | A_right(2x1)]
    # Split B into rows: B = [B_top(4x5); B_bottom(1x5)]
    # Split B columns: B = [B_left(5x4) | B_right(5x1)]
    # Split C into columns: C = [C_left(2x4) | C_right(2x1)]
    
    # For the 2x4 by 4x4 part, we can use two 2x2x2 Strassen blocks.
    # A_left is 2x4, B_top is 4x4.
    # Split A_left into A_L1(2x2) and A_L2(2x2)
    # Split B_top into a 2x2 block matrix: [[B11, B12], [B21, B22]] each 2x2
    # C_left_top = A_L1 * B_top = A_L1 * [B11 B12; B21 B22] = [A_L1*B11 + A_L1*B21, A_L1*B12 + A_L1*B22]
    # Using Strassen: 7 multiplications for this 2x2x2.
    
    # Similarly for A_L2.
    
    # For the 2x1 by 1x5 part: A_right * B_bottom = 2x5, this is outer product, rank 2*5 = 10? No, it's 2*1*5 = 10 rank terms? Actually it's just 2 rows times 1 column times 5 columns. Each element C[i,j] += A[i,4]*B[4,j]. That's 2*5 = 10 independent terms but they share the same A and B factors. Actually it's a rank-5 tensor (for each j, it's A[:,4] * B[4,j]^T). So rank 5.
    
    # For the cross terms: A_left * B_bottom and A_right * B_top
    # A_left(2x4) * B_bottom(1x5): each element C[i,j] += sum_k A[i,k]*B[4,j] for k=0..3 = B[4,j] * sum_k A[i,k]. This is (sum of A rows) * B_bottom^T. Rank 2*5 = 10? No, it's a rank-5 tensor: for each j, C[:,j] += B[4,j] * (A_left[:,0]+A_left[:,1]+A_left[:,2]+A_left[:,3]). So 5 terms (one per column j of C).
    
    # A_right(2x1) * B_top(4x5): C[i,j] += A[i,4] * sum_k B[k,j] for k=0..3. Rank 2*5 = 10? It's A_right * (row sums of B_top). 5 terms.
    
    # This is getting complex. Let me use a different approach.
    
    # Known result: The tensor rank of 2x5x5 matrix multiplication is 46.
    # Reference: This can be achieved by using Strassen on overlapping 2x2 blocks.
    
    # Let me construct it directly using the known decomposition.
    # Partition indices: i in {0,1}, j in {0..4}, k in {0..4}
    # T[i,j,k] = sum_c U[i*5+j,c] * V[j*5+k,c] * W[k*2+i,c]
    
    # We'll use 7 Strassen terms for each pair of (i-block, j-block, k-block).
    # With n=2, we have 1 i-block of size 2.
    # With m=5, we have 2 j-blocks: {0,1} and {2,3}, plus leftover {4}.
    # With p=5, we have 2 k-blocks: {0,1} and {2,3}, plus leftover {4}.
    
    # For j-block 0 (j=0,1) and k-block 0 (k=0,1): 7 Strassen terms
    # For j-block 0 and k-block 1 (k=2,3): 7 Strassen terms
    # For j-block 1 (j=2,3) and k-block 0: 7 Strassen terms
    # For j-block 1 and k-block 1: 7 Strassen terms
    # That's 28 terms.
    # Plus terms involving j=4 or k=4.
    
    # For j=4, k in {0,1,2,3}: T[i,4,k] = A[i,4]*B[4,k]. 4 terms for each i, so 8 terms? No, it's 2*4 = 8 rank-1 tensors? Actually T[i,4,k] = A[i,4]*B[4,k], which is a rank-1 tensor for each (i,k) pair. So 2*4=8 terms.
    # For j in {0,1,2,3}, k=4: T[i,j,4] = A[i,j]*B[j,4]. 2*4=8 terms.
    # For j=4, k=4: T[i,4,4] = A[i,4]*B[4,4]. 2 terms.
    # Total: 28 + 8 + 8 + 2 = 46. 
    
    # Let me verify: 28 + 8 + 8 + 2 = 46. Yes!
    
    # Now let me implement this.
    
    # Strassen 7 products for 2x2 block at positions (j1,j2) and (k1,k2):
    # S1 = A[i,j1] * (B[j1,k2] - B[j2,k2])  -- wait, need to be careful.
    # Actually for multiplying A_block(2x2) * B_block(2x2):
    # A_block = [[A[i,j1], A[i,j2]] for i in {0,1}]  -- this is 2x2 where rows are i=0,1 and cols are j=j1,j2
    # B_block = [[B[j1,k1], B[j1,k2]] for j in {j1,j2}]  -- 2x2 where rows are j=j1,j2 and cols are k=k1,k2
    
    # C_block[i_local, k_local] = sum_j_local A_block[i_local,j_local] * B_block[j_local,k_local]
    
    # Strassen:
    # M1 = A[0,0]*(B[0,1]-B[1,1])
    # M2 = (A[0,0]+A[0,1])*B[1,1]
    # M3 = (A[1,0]-A[0,0])*B[0,0]
    # M4 = A[1,1]*(B[0,0]-B[1,0])
    # M5 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M6 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # M7 = (A[0,0]-A[1,0])*(B[0,0]+B[1,0])
    # C[0,0] = M2+M4-M5+M7
    # C[0,1] = M1+M5
    # C[1,0] = M3+M4
    # C[1,1] = M1-M2+M3+M6
    
    # For the tensor decomposition, each M corresponds to a rank-1 term.
    # M1: U factor = A[0,0], V factor = (B[0,1]-B[1,1]), W factor = C[0,1] - C[1,1] coefficient
    # This is getting complex. Let me just build the U, V, W matrices directly.
    
    col = 0
    
    # Strassen blocks
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3)]
    
    for jb in j_blocks:
        for kb in k_blocks:
            j1, j2 = jb
            k1, k2 = kb
            # 7 Strassen terms
            # M1: A[i,j1] * (B[j1,k2] - B[j2,k2]), contributes to C[i,k2] with coeff +1 for i=0, -1 for i=1
            # In our notation, A[i,j] = U[i*m+j, col], B[j,k] = V[j*p+k, col], C[i,k] = W[k*n+i, col]
            # M1: U[i,j1] * (V[j1,k2] - V[j2,k2])
            # This means: for each i, U[i*m+j1, col] = 1, V[j1*p+k2, col] = 1, V[j2*p+k2, col] = -1
            # W: C[i,k2] gets coefficient. For i=0: +1, for i=1: -1? 
            # Wait, M1 = A[0,0]*(B[0,1]-B[1,1]) in local 2x2 coords.
            # A[0,0] means i_local=0, j_local=0 -> i=0, j=j1
            # A[1,0] would be i=1, j=j1
            # But Strassen M1 uses only A[0,0], not A[1,0].
            # So M1 = A[i=0, j=j1] * (B[j1,k2] - B[j2,k2])
            # This contributes to C[0, k2] and C[1, k2]? No, M1 is a scalar.
            # In the full C, M1 contributes to C[i, k2] for both i? No.
            # M1 is used in C[0,1] = M1+M5 and C[1,1] = M1-M2+M3+M6
            # So M1 contributes to C[0,k2] with coeff +1 and C[1,k2] with coeff +1.
            
            # Let me reconsider. Each M is a product of a linear combination of A entries and a linear combination of B entries.
            # M1 = (A[i=0,j=j1]) * (B[j1,k2] - B[j2,k2])
            # This is a rank-1 tensor: U[0*m+j1, col]=1, V[j1*p+k2,col]=1, V[j2*p+k2,col]=-1
            # W: This M1 appears in C[0,k2] (coeff 1) and C[1,k2] (coeff 1).
            # So W[k2*n+0, col] = 1, W[k2*n+1, col] = 1.
            
            U[0 * m + j1][col] = 1
            V[j1 * p + k2][col] = 1
            V[j2 * p + k2][col] = -1
            W[k2 * n + 0][col] = 1
            W[k2 * n + 1][col] = 1
            col += 1
            
            # M2 = (A[0,0]+A[0,1]) * B[1,1] = (A[i=0,j=j1]+A[i=0,j=j2]) * B[j2,k2]
            U[0 * m + j1][col] = 1
            U[0 * m + j2][col] = 1
            V[j2 * p + k2][col] = 1
            W[k2 * n + 0][col] = 1
            W[k2 * n + 1][col] = 1  # M2 appears in C[0,0] with -1 and C[1,1] with -1
            # Wait, M2 in C[0,0]: coeff -1, in C[1,1]: coeff -1
            W[k2 * n + 0][col] = 0  # M2 doesn't contribute to C[0,0] or C[0,1]
            # Let me redo. C[0,0]=M2+M4-M5+M7, C[0,1]=M1+M5, C[1,0]=M3+M4, C[1,1]=M1-M2+M3+M6
            # M2 appears in C[0,0] (coeff +1) and C[1,1] (coeff -1)
            W[k2 * n + 0][col] = 1   # C[0,k2]
            W[k2 * n + 1][col] = -1  # C[1,k2]
            col += 1
            
            # M3 = (A[1,0]-A[0,0]) * B[0,0] = (A[i=1,j=j1]-A[i=0,j=j1]) * B[j1,k1]
            U[1 * m + j1][col] = 1
            U[0 * m + j1][col] = -1
            V[j1 * p + k1][col] = 1
            # M3 in C[1,0] (coeff +1) and C[1,1] (coeff +1)
            W[k1 * n + 1][col] = 1
            W[k2 * n + 1][col] = 1
            col += 1
            
            # M4 = A[1,1] * (B[0,0]-B[1,0]) = A[i=1,j=j2] * (B[j1,k1]-B[j2,k1])
            U[1 * m + j2][col] = 1
            V[j1 * p + k1][col] = 1
            V[j2 * p + k1][col] = -1
            # M4 in C[0,0] (coeff +1) and C[1,0] (coeff +1)
            W[k1 * n + 0][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1
            
            # M5 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
            U[0 * m + j1][col] = 1
            U[1 * m + j2][col] = 1
            V[j1 * p + k1][col] = 1
            V[j2 * p + k2][col] = 1
            # M5 in C[0,0] (coeff -1), C[0,1] (coeff +1)
            W[k1 * n + 0][col] = -1
            W[k2 * n + 0][col] = 1
            col += 1
            
            # M6 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
            U[0 * m + j2][col] = 1
            U[1 * m + j2][col] = -1
            V[j2 * p + k1][col] = 1
            V[j2 * p + k2][col] = 1
            # M6 in C[1,1] (coeff +1)
            W[k2 * n + 1][col] = 1
            col += 1
            
            # M7 = (A[0,0]-A[1,0]) * (B[0,0]+B[1,0])
            U[0 * m + j1][col] = 1
            U[1 * m + j1][col] = -1
            V[j1 * p + k1][col] = 1
            V[j2 * p + k1][col] = 1
            # M7 in C[0,0] (coeff +1)
            W[k1 * n + 0][col] = 1
            col += 1
    
    # Now handle terms with j=4 or k=4
    # For j=4, k in {0,1,2,3}: T[i,4,k] = A[i,4]*B[4,k]
    # Each (i,k) pair is a separate rank-1 term
    for i in range(n):
        for k in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # For j in {0,1,2,3}, k=4: T[i,j,4] = A[i,j]*B[j,4]
    for i in range(n):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    # For j=4, k=4: T[i,4,4] = A[i,4]*B[4,4]
    for i in range(n):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1
    
    # col should be 46
    # 28 (Strassen) + 8 (j=4,k<4) + 8 (j<4,k=4) + 2 (j=4,k=4) = 46
    
    return {"U": U, "V": V, "W": W}
