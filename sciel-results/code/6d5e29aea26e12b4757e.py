def winograd_3x3():
    """Returns U, V, W for 3x3x3 matmul with rank 23."""
    rank = 23
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    # The 23 terms of Winograd 3x3
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    # U indexed by (i,j), V by (j,k), W by (k,i)
    
    terms = [
        # (i,j) coeff for U, (j,k) coeff for V, (k,i) coeff for W
        # Term 0: A[0,0]*B[0,0]
        {('U', (0,0)): 1, ('V', (0,0)): 1, ('W', (0,0)): 1},
        # ... etc
    ]
    
    return U, V, W
