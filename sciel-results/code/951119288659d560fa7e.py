"""Matrix multiplication 2x3x10 tensor decomposition."""

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    # Each 2x3x2 block has rank 11 (7 Strassen + 4 standard for 3rd row)
    # 5 blocks => rank 55
    rank = 55
    
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0.0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0.0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    for t in range(5):
        # Strassen rank-7 for 2x2x2 (rows 0,1 of A; cols 0,1 of B)
        # M1 = A[0][0]*B[0][0]
        # M2 = A[1][0]*B[0][0]
        # M3 = A[0][0]*B[1][1]
        # M4 = A[1][1]*B[0][1]
        # M5 = (A[0][0]+A[0][1])*B[1][1]
        # M6 = (A[1][0]-A[0][0])*(B[0][0]+B[0][1])
        # M7 = (A[0][1]-A[1][1])*(B[1][0]+B[1][1])
        
        # B indices: B row r, col c -> index r*10 + (2*t + c)
        b00 = 2 * t        # B[0][2t]
        b01 = 2 * t + 1    # B[0][2t+1]
        b10 = 10 + 2 * t   # B[1][2t]
        b11 = 10 + 2 * t + 1  # B[1][2t+1]
        b20 = 20 + 2 * t   # B[2][2t]
        b21 = 20 + 2 * t + 1  # B[2][2t+1]
        
        # A indices: A[i][j] -> i*3+j
        a00 = 0  # A[0][0]
        a01 = 1  # A[0][1]
        a02 = 2  # A[0][2]
        a10 = 3  # A[1][0]
        a11 = 4  # A[1][1]
        a12 = 5  # A[1][2]
        
        # W indices: W[k*n+i, r] where k is col of C, i is row of C
        # For block t, C cols are 2t and 2t+1
        w_c0i0 = (2*t) * n + 0     # C[2t][0]
        w_c1i0 = (2*t+1) * n + 0   # C[2t+1][0]
        w_c0i1 = (2*t) * n + 1     # C[2t][1]
        w_c1i1 = (2*t+1) * n + 1   # C[2t+1][1]
        
        # Strassen rank-7
        # M1: A[0][0] * B[0][0] -> C[0][0] of block
        U[a00][col] = 1.0
        V[b00][col] = 1.0
        W[w_c0i0][col] = 1.0
        col += 1
        
        # M2: A[1][0] * B[0][0] -> C[1][0] of block
        U[a10][col] = 1.0
        V[b00][col] = 1.0
        W[w_c0i1][col] = 1.0
        col += 1
        
        # M3: A[0][0] * B[1][1] -> C[0][1] of block
        U[a00][col] = 1.0
        V[b11][col] = 1.0
        W[w_c1i0][col] = 1.0
        col += 1
        
        # M4: A[1][1] * B[0][1] -> C[1][1] of block
        U[a11][col] = 1.0
        V[b01][col] = 1.0
        W[w_c1i1][col] = 1.0
        col += 1
        
        # M5: (A[0][0]+A[0][1]) * B[1][1] -> C[0][1] of block
        U[a00][col] = 1.0
        U[a01][col] = 1.0
        V[b11][col] = 1.0
        W[w_c1i0][col] = 1.0
        col += 1
        
        # M6: (A[1][0]-A[0][0]) * (B[0][0]+B[0][1]) -> C[1][0] and C[1][1] of block
        U[a10][col] = 1.0
        U[a00][col] = -1.0
        V[b00][col] = 1.0
        V[b01][col] = 1.0
        W[w_c0i1][col] = 1.0
        W[w_c1i1][col] = 1.0
        col += 1
        
        # M7: (A[0][1]-A[1][1]) * (B[1][0]+B[1][1]) -> C[0][0] and C[0][1] of block
        U[a01][col] = 1.0
        U[a11][col] = -1.0
        V[b10][col] = 1.0
        V[b11][col] = 1.0
        W[w_c0i0][col] = 1.0
        W[w_c1i0][col] = 1.0
        col += 1
        
        # Standard 4 for row 2 of A with both cols of B
        # M8: A[0][2] * B[2][2t] -> C[0][2t]
        U[a02][col] = 1.0
        V[b20][col] = 1.0
        W[w_c0i0][col] = 1.0
        col += 1
        
        # M9: A[0][2] * B[2][2t+1] -> C[0][2t+1]
        U[a02][col] = 1.0
        V[b21][col] = 1.0
        W[w_c1i0][col] = 1.0
        col += 1
        
        # M10: A[1][2] * B[2][2t] -> C[1][2t]
        U[a12][col] = 1.0
        V[b20][col] = 1.0
        W[w_c0i1][col] = 1.0
        col += 1
        
        # M11: A[1][2] * B[2][2t+1] -> C[1][2t+1]
        U[a12][col] = 1.0
        V[b21][col] = 1.0
        W[w_c1i1][col] = 1.0
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
