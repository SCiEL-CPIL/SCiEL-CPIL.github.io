"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-49 decomposition for 4x4 matrix multiplication using the 
    Kronecker product of the Strassen algorithm for 2x2 matrices.
    """
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2 matrices
    # Indices: i, j in {0, 1}
    # U maps A[i, j] to factor indices
    # V maps B[j, k] to factor indices
    # W maps C[i, k] to factor indices (Note: Task uses W[k*n+i, r] for C[i,k])
    
    # There are 7 multiplications in Strassen. Let's denote them 0..6.
    # We define the coefficients for U, V, W for the 2x2 case.
    # U[i*2+j, c] = coeff_A
    # V[j*2+k, c] = coeff_B
    # W[k*2+i, c] = coeff_C  (Note the index swap in W definition vs standard C[i,k])
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Multiplication 0: M0 = A00 * (B00 - B11)
    U2[0][0] = 1   # A00
    V2[0][0] = 1   # B00
    V2[3][0] = -1  # B11
    
    # Multiplication 1: M1 = (A00 + A10) * B11
    U2[0][1] = 1   # A00
    U2[2][1] = 1   # A10
    V2[3][1] = 1   # B11
    
    # Multiplication 2: M2 = (A01 - A11) * (B10 + B11)
    U2[1][2] = 1   # A01
    U2[3][2] = -1  # A11
    V2[2][2] = 1   # B10
    V2[3][2] = 1   # B11
    
    # Multiplication 3: M3 = A11 * (B01 - B00)
    U2[3][3] = 1   # A11
    V2[1][3] = 1   # B01
    V2[0][3] = -1  # B00
    
    # Multiplication 4: M4 = (A00 + A11) * B00
    U2[0][4] = 1   # A00
    U2[3][4] = 1   # A11
    V2[0][4] = 1   # B00
    
    # Multiplication 5: M5 = (A01 - A00) * (B00 + B10)
    U2[1][5] = 1   # A01
    U2[0][5] = -1  # A00
    V2[0][5] = 1   # B00
    V2[2][5] = 1   # B10
    
    # Multiplication 6: M6 = (A10 - A11) * (B11 + B01)
    U2[2][6] = 1   # A10
    U2[3][6] = -1  # A11
    V2[3][6] = 1   # B11
    V2[1][6] = 1   # B01
    
    # Output C for 2x2:
    # C00 = M4 + M3 - M1 + M5
    # C01 = M0 + M1
    # C10 = M2 + M3
    # C11 = M4 - M0 + M2 + M6
    
    # W2[k*2+i, c] corresponds to C[i,k]
    # C00 (i=0, k=0) -> idx 0*2+0 = 0
    W2[0][4] = 1
    W2[0][3] = 1
    W2[0][1] = -1
    W2[0][5] = 1
    
    # C01 (i=0, k=1) -> idx 1*2+0 = 2
    W2[2][0] = 1
    W2[2][1] = 1
    
    # C10 (i=1, k=0) -> idx 0*2+1 = 1
    W2[1][2] = 1
    W2[1][3] = 1
    
    # C11 (i=1, k=1) -> idx 1*2+1 = 3
    W2[3][4] = 1
    W2[3][0] = -1
    W2[3][2] = 1
    W2[3][6] = 1

    # Now construct 4x4 factors using Kronecker product
    # Rank = 7 * 7 = 49
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Indices for 4x4:
    # i4 = i2 * 2 + i1
    # j4 = j2 * 2 + j1
    # k4 = k2 * 2 + k1
    # c4 = c2 * 7 + c1
    
    for i2 in range(2):
        for i1 in range(2):
            i4 = i2 * 2 + i1
            for j2 in range(2):
                for j1 in range(2):
                    j4 = j2 * 2 + j1
                    for k2 in range(2):
                        for k1 in range(2):
                            k4 = k2 * 2 + k1
                            
                            # Sum over c2, c1
                            # U4[i4, c4] = sum U2[i2*2+j2, c2] * U2[i1*2+j1, c1]
                            # V4[j4, c4] = sum V2[j2*2+k2, c2] * V2[j1*2+k1, c1]
                            # W4[k4, c4] = sum W2[k2*2+i2, c2] * W2[k1*2+i1, c1]
                            
                            for c2 in range(7):
                                for c1 in range(7):
                                    c4 = c2 * 7 + c1
                                    
                                    u_val = U2[i2 * 2 + j2][c2] * U2[i1 * 2 + j1][c1]
                                    v_val = V2[j2 * 2 + k2][c2] * V2[j1 * 2 + k1][c1]
                                    w_val = W2[k2 * 2 + i2][c2] * W2[k1 * 2 + i1][c1]
                                    
                                    U[i4][c4] += u_val
                                    V[j4][c4] += v_val
                                    W[k4][c4] += w_val
                                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
