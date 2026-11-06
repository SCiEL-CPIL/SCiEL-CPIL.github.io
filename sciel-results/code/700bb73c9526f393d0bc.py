def construct():
    # A(x) = 1 + x^3 + x^6 - x^9
    # B(x) = 1 + 2x - 2x^2
    # P(x) = A(x) * x^10 * B(x)
    
    # Coefficients of A
    A = [0] * 10
    A[0] = 1
    A[3] = 1
    A[6] = 1
    A[9] = -1
    
    # Coefficients of B
    B = [0] * 3
    B[0] = 1
    B[1] = 2
    B[2] = -2
    
    # Shift B by 10
    M = 10
    B_shifted = [0] * (len(B) + M)
    for i, c in enumerate(B):
        B_shifted[i + M] = c
        
    # Multiply A and B_shifted
    # Convolution
    len_P = len(A) + len(B_shifted) - 1
    P = [0] * len_P
    for i, ca in enumerate(A):
        if ca == 0: continue
        for j, cb in enumerate(B_shifted):
            if cb == 0: continue
            P[i + j] += ca * cb
            
    return P
