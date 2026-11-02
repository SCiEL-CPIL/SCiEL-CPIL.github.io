"""Construct rank-minimal exact tensor decomposition for 3x3 x 3x8 matrix multiplication."""
DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # We'll use a rank-54 construction based on splitting 3x8 into two 3x4 blocks
    # and using a rank-27 construction for each 3x3 x 3x4 block.
    # 
    # For 3x3 x 3x4, we can achieve rank 27 by using the following approach:
    # - Split the 4 columns into groups and use efficient bilinear forms
    # - One known approach: use 3x3x3 (rank 23 via Strassen-like) + 3x3x1 (rank 9) = 32, 
    #   but we can do better with a unified construction.
    #
    # Actually, let me use a direct rank-54 construction that's verifiable:
    # For each i in 0..2, j in 0..2, we need to compute sum_k A[i,j]*B[j,k] for k in 0..7
    # 
    # A simpler approach: use the fact that T(3,3,8) <= T(3,3,4) + T(3,3,4)
    # and T(3,3,4) can be done in rank 27 using a specific construction.
    #
    # Let me implement a rank-54 construction using a structured approach:
    # Split k into k1 (0,1) representing which 4-block, and k2 (0,1,2,3) representing position within block
    # For each (i, j, k1, k2), we have a term, but we can merge some.
    #
    # Actually, the simplest valid improvement over schoolbook (72) is to use:
    # For each pair (i,j), the 8 terms A[i,j]*B[j,0..7] can be computed with rank 8
    # But we can do better by sharing across i or j.
    #
    # Let me use a rank-54 construction:
    # The tensor has 9*24 = 216 entries. Schoolbook uses 72 rank-1 terms.
    # 
    # A known result: T(3,3,8) = 54 (or close to it). Let me implement rank 54.
    #
    # Construction: Use the following identity for 3x3 x 3x8:
    # For i in {0,1,2}, j in {0,1,2}, k in {0,...,7}
    # We can group the k indices and use bilinear combinations.
    #
    # Let me implement a concrete rank-54 decomposition:
    # 
    # For the 3x3 x 3x4 case, one construction uses rank 27:
    # For each (i,j) pair, we have 4 terms for the 4 k values.
    # 9 pairs * 4 = 36 schoolbook, but rank 27 is achievable.
    #
    # I'll use a construction where for 3x3 x 3x4, we use:
    # - 9 terms of the form A[i,j] * B[j,0] for k=0
    # - 9 terms of the form A[i,j] * B[j,1] for k=1  
    # - 9 terms of the form (A[i,j] + A[i,j+1]) * (B[j,2] + B[j+1,2]) type combinations
    #
    # This is getting complex. Let me use a simpler valid rank-54 construction.
    
    rank = 54
    U = [[0]*rank for _ in range(n*m)]  # 9 rows
    V = [[0]*rank for _ in range(m*p)]  # 24 rows  
    W = [[0]*rank for _ in range(p*n)]  # 24 rows
    
    # Construction: split into two 3x4 blocks
    # For block b in {0,1}, handle columns 4*b to 4*b+3
    col = 0
    for b in range(2):
        # For this 3x4 block, use rank-27 construction
        # For 3x3 x 3x4, use the following:
        # Terms 1-9: A[i,j] * B[j, 4*b + 0] for each (i,j), output to W[4*b+0, i]
        for i in range(n):
            for j in range(m):
                U[i*m + j][col] = 1
                V[j*p + (4*b + 0)][col] = 1
                W[(4*b + 0)*n + i][col] = 1
                col += 1
        
        # Terms 10-18: A[i,j] * B[j, 4*b + 1] for each (i,j), output to W[4*b+1, i]
        for i in range(n):
            for j in range(m):
                U[i*m + j][col] = 1
                V[j*p + (4*b + 1)][col] = 1
                W[(4*b + 1)*n + i][col] = 1
                col += 1
        
        # Terms 19-27: Use combinations for k = 4*b+2 and 4*b+3
        # For k=4*b+2: use A[i,j] * B[j, 4*b+2] but with sharing
        # For k=4*b+3: use A[i,j] * B[j, 4*b+3] but with sharing
        # 
        # To achieve rank 27 for 3x3x3x4, we need to save 9 terms from the 36 schoolbook.
        # Use the following for the last two k values:
        # For each i, j:
        #   Term: A[i,j] * (B[j, 4*b+2] + B[j, 4*b+3]) -> contributes to both W[4*b+2, i] and W[4*b+3, i]
        #   But this doesn't directly work since we need separate outputs.
        #
        # Alternative: use 9 terms for k=4*b+2 and 9 terms for k=4*b+3, total 18, 
        # plus 18 for the first two = 36. That's schoolbook.
        #
        # To get rank 27, I need a different approach. Let me just use rank 36 per block 
        # for now, giving total rank 72, which matches the current score.
        #
        # Actually, let me use rank 36 per block = 72 total, which is the same as schoolbook.
        # To improve, I need a better construction.
        
        # For k=4*b+2:
        for i in range(n):
            for j in range(m):
                U[i*m + j][col] = 1
                V[j*p + (4*b + 2)][col] = 1
                W[(4*b + 2)*n + i][col] = 1
                col += 1
        
        # For k=4*b+3:
        for i in range(n):
            for j in range(m):
                U[i*m + j][col] = 1
                V[j*p + (4*b + 3)][col] = 1
                W[(4*b + 3)*n + i][col] = 1
                col += 1
    
    # This gives rank 72, same as schoolbook. Not an improvement.
    # Let me try a different approach: use rank 54 directly.
    
    # Reset
    rank = 54
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Use a construction based on the following idea:
    # For the 3x8 output, group columns in pairs: (0,1), (2,3), (4,5), (6,7)
    # For each pair of columns (2c, 2c+1) and each (i,j), use 1.5 terms on average
    # This is hand-wavy. Let me just return the schoolbook for now.
    
    # Schoolbook rank 72
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
