<<<<<<< SEARCH
    # Block index groups: 2+2+1 partition
    blocks = [[0, 1], [2, 3], [4]]
    
    # Standard Strassen 2x2x2 terms for local indices 0,1
    # Returns list of (u_coeffs, v_coeffs, w_coeffs)
    STRASSEN_2x2_TERMS = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
=======
    # Block index groups: 2+2+1 partition
    blocks = [[0, 1], [2, 3], [4]]
    
    # Standard Strassen 2x2x2 terms for local indices 0,1
    # Returns list of (u_coeffs, v_coeffs, w_coeffs)
    STRASSEN_2x2_TERMS = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # For 2x2x1 blocks, use a rank-3 decomposition instead of schoolbook rank-4
    # This exploits the fact that we only need to compute 4 products but can do it with 3 multiplications
    # using the identity: (a+b)(c+d) - ac - bd = ad + bc
    # For 2x2x1: C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, j in {0,1}, k fixed
    # We can use: (A[i0,j0]+A[i1,j1])*(B[j0,k]+B[j1,k]) - A[i0,j0]*B[j0,k] - A[i1,j1]*B[j1,k]
    # This gives us 3 terms instead of 4
    STRASSEN_2x2x1_TERMS = [
        # Term 1: (A[0,0]+A[1,1]) * (B[0,k]+B[1,k]) contributes to C[0,k] and C[1,k]
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,0): 1}, {(0,0): 1, (1,0): 1}),
        # Term 2: -A[0,0] * B[0,k] 
        ({(0,0): 1}, {(0,0): 1}, {(0,0): -1}),
        # Term 3: -A[1,1] * B[1,k]
        ({(1,1): 1}, {(1,0): 1}, {(1,0): -1}),
    ]
    
    # For 2x1x2 blocks (2 rows in A, 1 col in B, 2 cols in C)
    # C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, j fixed, k in {0,1}
    # Use: (A[i0,j]+A[i1,j])*(B[j,0]+B[j,1]) - A[i0,j]*B[j,0] - A[i1,j]*B[j,1]
    STRASSEN_2x1x2_TERMS = [
        # Term 1: (A[0,j]+A[1,j]) * (B[j,0]+B[j,1])
        ({(0,0): 1, (1,0): 1}, {(0,0): 1, (0,1): 1}, {(0,0): 1, (0,1): 1}),
        # Term 2: -A[0,j] * B[j,0]
        ({(0,0): 1}, {(0,0): 1}, {(0,0): -1}),
        # Term 3: -A[1,j] * B[j,1]
        ({(1,0): 1}, {(0,1): 1}, {(0,1): -1}),
    ]
    
    # For 1x2x2 blocks (1 row in A, 2 cols in B, 2 cols in C)
    # C[i,k] = sum_j A[i,j]*B[j,k] for i fixed, j in {0,1}, k in {0,1}
    # Use: (A[i,0]+A[i,1])*(B[0,0]+B[1,1]) - A[i,0]*B[0,0] - A[i,1]*B[1,1]
    STRASSEN_1x2x2_TERMS = [
        # Term 1: (A[i,0]+A[i,1]) * (B[0,0]+B[1,1])
        ({(0,0): 1, (0,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1}),
        # Term 2: -A[i,0] * B[0,0]
        ({(0,0): 1}, {(0,0): 1}, {(0,0): -1}),
        # Term 3: -A[i,1] * B[1,1]
        ({(0,1): 1}, {(1,1): 1}, {(0,1): -1}),
    ]
>>>>>>> REPLACE

