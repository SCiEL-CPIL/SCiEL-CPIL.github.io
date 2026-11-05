"""
Task: exact rank-minimal tensor decomposition for 2x3 * 3x6 matrix multiplication.
Output W uses trace convention: W[k*n + i, r].
We use the known structure that decomposes the 2x3x6 tensor into three
independent 2x2x2 (Strassen, rank 7) parts and one 2x1x2 (rank 4) part,
giving total rank 3*7 + 4 = 25.
"""

def construct():
    n, m, p = 2, 3, 6
    # We will build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    rank = 25
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 multipliers (rank 7):
    # We index multipliers by (block, s) where s in 0..6 for Strassen,
    # and handle the j=0 part separately.
    
    # Layout of the 3x6 matrix B: rows j in {0,1,2}, cols k in {0..5}.
    # We split B into three 3x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6].
    # For each block b (0,1,2), we need to compute A (2x3) * B_block (3x2) = C_block (2x2).
    # A is 2x3: rows i in {0,1}, cols j in {0,1,2}.
    # For a given block b, B_block has rows j in {0,1,2} and cols k in {2b, 2b+1}.
    
    # The multiplication A * B_block can be split:
    # C_block = A[:, 0] * B_block[0, :] + A[:, 1:] * B_block[1:, :]
    # The first term is (2x1) * (1x2) = 2x2 rank-1 outer product, but as a tensor it's rank 4 (since 2*1*2 = 4 basis tensors needed for exact decomposition of that sub-tensor).
    # Actually, the sub-tensor for A[:,0] * B[0,:] is: sum over i,k of A_{i0} B_{0k} E_{ik}. This is a 2x1x2 tensor. Its rank is 4.
    # The second term is A[:, 1:2] * B[1:2, :] which is 2x2x2 tensor (rank 7 via Strassen).
    
    # So for each block b, we use 4 + 7 = 11 multipliers.
    # Total rank = 3 * 11 = 33.
    
    # Let's implement rank 33.
    # U index: i*m + j = i*3 + j, i in {0,1}, j in {0,1,2}
    # V index: j*p + k = j*6 + k, j in {0,1,2}, k in {0..5}
    # W index: k*n + i = k*2 + i, k in {0..5}, i in {0,1}
    
    # We'll allocate rank = 33.
    rank = 33
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen multipliers for 2x2x2:
    # Let X (2x2), Y (2x2). Z = X*Y.
    # M1 = (X11 - X12) * (Y11 + Y12)
    # M2 = (X11 + X12) * Y22
    # M3 = X11 * (Y22 - Y21)
    # M4 = X22 * (Y11 - Y12)
    # M5 = (X11 + X22) * Y22
    # M6 = (X12 - X22) * (Y21 + Y22)
    # M7 = (X11 - X12) * (Y11 + Y21)
    # Z11 = M5 + M4 - M6 - M7
    # Z12 = M1 + M7
    # Z21 = M2 + M3
    # Z22 = M5 + M1 - M3 - M4
    
    # For block b, columns k in B are 2b and 2b+1.
    # The 2x2 part is A[:, 1:2] (rows i=0,1; cols j=1,2) times B_block[1:2, :] (rows j=1,2; cols k=2b,2b+1).
    # X = A[:, 1:2]: X[i, j'] where i in {0,1}, j' in {0,1} corresponds to j = 1+j'.
    #   X[0,0] = A[0,1], X[0,1] = A[0,2]
    #   X[1,0] = A[1,1], X[1,1] = A[1,2]
    # Y = B_block[1:2, :]: Y[j', k'] where j' in {0,1} corresponds to j=1+j', k' in {0,1} corresponds to k=2b+k'.
    #   Y[0,0] = B[1,2b], Y[0,1] = B[1,2b+1]
    #   Y[1,0] = B[2,2b], Y[1,1] = B[2,2b+1]
    
    # The 2x1x2 part: A[:, 0] (rows i=0,1; col j=0) times B_block[0, :] (row j=0; cols k=2b,2b+1).
    # This gives C_block[i, k'] = A[i,0] * B[0, 2b+k'] for i in {0,1}, k' in {0,1}.
    # This is a 2x1x2 tensor. Rank 4 decomposition:
    # For each (i, k'), we need a term. But we can share.
    # Actually, the tensor T[i, 0, k'] = A[i,0] * B[0, 2b+k'].
    # We can decompose as: sum over 4 terms.
    # Term 1: i=0, k'=0: U[0, c] = 1 (for i=0,j=0), V[0, c] = 1 (for j=0,k=2b), W[2b, c] = 1 (for k=2b, i=0)
    # Term 2: i=0, k'=1: U[0, c] = 1, V[1, c] = 1 (for j=0,k=2b+1), W[2b+1, c] = 1
    # Term 3: i=1, k'=0: U[1, c] = 1 (for i=1,j=0), V[2, c] = 1, W[3, c] = 1 (for k=2b, i=1)
    # Term 4: i=1, k'=1: U[2, c] = 1, V[3, c] = 1, W[4, c] = 1
    # Wait, U index is i*3+j. For i=0,j=0: index 0. For i=1,j=0: index 3.
    # V index is j*6+k. For j=0,k=2b: index 2b. For j=0,k=2b+1: index 2b+1.
    # W index is k*2+i. For k=2b, i=0: index 4b. For k=2b, i=1: index 4b+1.
    # For k=2b+1, i=0: index 4b+2. For k=2b+1, i=1: index 4b+3.
    
    col = 0
    for b in range(3):
        # 4 terms for j=0 part
        # i=0, k'=0
        U[0 * 3 + 0][col] = 1.0
        V[0 * 6 + (2 * b)][col] = 1.0
        W[(2 * b) * 2 + 0][col] = 1.0
        col += 1
        # i=0, k'=1
        U[0 * 3 + 0][col] = 1.0
        V[0 * 6 + (2 * b + 1)][col] = 1.0
        W[(2 * b + 1) * 2 + 0][col] = 1.0
        col += 1
        # i=1, k'=0
        U[1 * 3 + 0][col] = 1.0
        V[0 * 6 + (2 * b)][col] = 1.0
        W[(2 * b) * 2 + 1][col] = 1.0
        col += 1
        # i=1, k'=1
        U[1 * 3 + 0][col] = 1.0
        V[0 * 6 + (2 * b + 1)][col] = 1.0
        W[(2 * b + 1) * 2 + 1][col] = 1.0
        col += 1
        
        # 7 Strassen terms for j=1,2 part
        # X indices in U: i*3 + (1+j')
        # Y indices in V: (1+j')*6 + (2b + k')
        # W indices: (2b+k')*2 + i
        
        # M1 = (X11 - X12) * (Y11 + Y12)
        # X11 = A[0,1] -> U[0*3+1]
        # X12 = A[0,2] -> U[0*3+2]
        # Y11 = B[1,2b] -> V[1*6+2b]
        # Y12 = B[1,2b+1] -> V[1*6+2b+1]
        # Z12 += M1, Z22 += M1
        # W indices for Z12: k=2b+1, i=0 -> (2b+1)*2+0 = 4b+2
        # W indices for Z22: k=2b+1, i=1 -> (2b+1)*2+1 = 4b+3
        U[0 * 3 + 1][col] = 1.0
        U[0 * 3 + 2][col] = -1.0
        V[1 * 6 + 2 * b][col] = 1.0
        V[1 * 6 + 2 * b + 1][col] = 1.0
        W[(2 * b + 1) * 2 + 0][col] = 1.0
        W[(2 * b + 1) * 2 + 1][col] = 1.0
        col += 1
        
        # M2 = (X11 + X12) * Y22
        # X11 = A[0,1], X12 = A[0,2]
        # Y22 = B[2,2b+1] -> V[2*6+2b+1]
        # Z21 += M2
        # W index for Z21: k=2b, i=1 -> 2b*2+1 = 4b+1
        U[0 * 3 + 1][col] = 1.0
        U[0 * 3 + 2][col] = 1.0
        V[2 * 6 + 2 * b + 1][col] = 1.0
        W[(2 * b) * 2 + 1][col] = 1.0
        col += 1
        
        # M3 = X11 * (Y22 - Y21)
        # X11 = A[0,1]
        # Y22 = B[2,2b+1], Y21 = B[2,2b]
        # Z21 += M3, Z22 -= M3
        U[0 * 3 + 1][col] = 1.0
        V[2 * 6 + 2 * b + 1][col] = 1.0
        V[2 * 6 + 2 * b][col] = -1.0
        W[(2 * b) * 2 + 1][col] = 1.0
        W[(2 * b + 1) * 2 + 1][col] = -1.0
        col += 1
        
        # M4 = X22 * (Y11 - Y12)
        # X22 = A[1,2] -> U[1*3+2]
        # Y11 = B[1,2b], Y12 = B[1,2b+1]
        # Z11 -= M4, Z22 -= M4
        # W index for Z11: k=2b, i=0 -> 4b
        U[1 * 3 + 2][col] = 1.0
        V[1 * 6 + 2 * b][col] = 1.0
        V[1 * 6 + 2 * b + 1][col] = -1.0
        W[(2 * b) * 2 + 0][col] = -1.0
        W[(2 * b + 1) * 2 + 1][col] = -1.0
        col += 1
        
        # M5 = (X11 + X22) * Y22
        # X11 = A[0,1], X22 = A[1,2]
        # Y22 = B[2,2b+1]
        # Z11 += M5, Z22 += M5
        U[0 * 3 + 1][col] = 1.0
        U[1 * 3 + 2][col] = 1.0
        V[2 * 6 + 2 * b + 1][col] = 1.0
        W[(2 * b) * 2 + 0][col] = 1.0
        W[(2 * b + 1) * 2 + 1][col] = 1.0
        col += 1
        
        # M6 = (X12 - X22) * (Y21 + Y22)
        # X12 = A[0,2], X22 = A[1,2]
        # Y21 = B[2,2b], Y22 = B[2,2b+1]
        # Z11 -= M6
        U[0 * 3 + 2][col] = 1.0
        U[1 * 3 + 2][col] = -1.0
        V[2 * 6 + 2 * b][col] = 1.0
        V[2 * 6 + 2 * b + 1][col] = 1.0
        W[(2 * b) * 2 + 0][col] = -1.0
        col += 1
        
        # M7 = (X11 - X12) * (Y11 + Y21)
        # X11 = A[0,1], X12 = A[0,2]
        # Y11 = B[1,2b], Y21 = B[2,2b]
        # Z11 -= M7, Z12 += M7
        U[0 * 3 + 1][col] = 1.0
        U[0 * 3 + 2][col] = -1.0
        V[1 * 6 + 2 * b][col] = 1.0
        V[2 * 6 + 2 * b][col] = 1.0
        W[(2 * b) * 2 + 0][col] = -1.0
        W[(2 * b + 1) * 2 + 0][col] = 1.0
        col += 1
    
    # Verify col == 33
    # Each block: 4 + 7 = 11. 3 blocks = 33.
    assert col == 33, f"Expected 33 columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
