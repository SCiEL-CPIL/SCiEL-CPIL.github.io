"""Tensor decomposition for 2x3x3 matrix multiplication with rank 17."""
DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = 2, 3, 3
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 17
    V = [[0] * rank for _ in range(m * p)]  # 9 x 17
    W = [[0] * rank for _ in range(p * n)]  # 6 x 17
    
    col = 0
    
    # Part 1: A1*B11 (2x2 * 2x2) using Strassen, rank 7
    # A1: U[0]=A[0,0], U[1]=A[0,1], U[3]=A[1,0], U[4]=A[1,1]
    # B11: V[0]=B[0,0], V[1]=B[0,1], V[3]=B[1,0], V[4]=B[1,1]
    # C11: W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
    
    a1_idx = [0, 1, 3, 4]  # A11, A12, A21, A22
    b11_idx = [0, 1, 3, 4]  # B11, B12, B21, B22
    c11_idx = [0, 1, 2, 3]  # C11, C12, C21, C22
    
    # Strassen: 7 multiplications M1..M7
    # U factors (coefficients for A11, A12, A21, A22):
    strassen_U = [
        [1, 0, 0, 1],   # M1: (A11+A22)
        [0, 0, 1, 1],   # M2: (A21+A22)
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: (A11+A12)
        [-1, 0, 1, 0],  # M6: (A21-A11)
        [0, 1, 0, -1],  # M7: (A12-A22)
    ]
    
    # V factors (coefficients for B11, B12, B21, B22):
    strassen_V = [
        [0, 1, 1, 0],   # M1: (B12+B21)
        [1, 0, 0, 0],   # M2: B11
        [0, 0, 1, -1],  # M3: (B21-B22)
        [1, 0, -1, 0],  # M4: (B11-B21)
        [0, 0, 0, 1],   # M5: B22
        [1, 1, 0, 0],   # M6: (B11+B12)
        [0, 0, 1, 1],   # M7: (B21+B22)
    ]
    
    # W factors (coefficients for C11, C12, C21, C22):
    strassen_W = [
        [1, 0, 0, 1],    # C11: M1 + M4 - M5 + M7 -> [1,0,0,1,-1,0,1]
        [0, 1, 1, 0],    # C12: M3 + M5 -> [0,0,1,0,1,0,0]
        [0, 0, 1, 1],    # C21: M2 + M4 -> [0,1,0,1,0,0,0]
        [1, 0, 1, 0],    # C22: M1 - M2 + M3 + M6 -> [1,-1,1,0,0,1,0]
    ]
    
    # Wait, I need to index strassen_W by output position and multiply by M index
    # Let me redo: for each M_s, add to W[c11_idx[ci]] the coefficient
    
    # For each Strassen multiplication s=0..6:
    for s in range(7):
        # Add U contributions
        for ai in range(4):
            if strassen_U[s][ai] != 0:
                U[a1_idx[ai]][col] += strassen_U[s][ai]
        # Add V contributions
        for bi in range(4):
            if strassen_V[s][bi] != 0:
                V[b11_idx[bi]][col] += strassen_V[s][bi]
        # Add W contributions: for each output C_c, coefficient from M_s
        # C11: M1(+1), M4(+1), M5(-1), M7(+1) => s=0:+1, s=3:+1, s=4:-1, s=6:+1
        # C12: M3(+1), M5(+1) => s=2:+1, s=4:+1
        # C21: M2(+1), M4(+1) => s=1:+1, s=3:+1
        # C22: M1(+1), M2(-1), M3(+1), M6(+1) => s=0:+1, s=1:-1, s=2:+1, s=5:+1
        
        w_coeffs = {
            0: [1, 0, 0, 1, -1, 0, 1],  # C11
            1: [0, 0, 1, 0, 1, 0, 0],   # C12
            2: [0, 1, 0, 1, 0, 0, 0],   # C21
            3: [1, -1, 1, 0, 0, 1, 0],  # C22
        }
        for ci in range(4):
            coeff = w_coeffs[ci][s]
            if coeff != 0:
                W[c11_idx[ci]][col] += coeff
        col += 1
    
    # Part 2: A1*B12 (2x2 * 2x1), schoolbook rank 4
    # A1: U[0]=A[0,0], U[1]=A[0,1], U[3]=A[1,0], U[4]=A[1,1]
    # B12: V[2]=B[0,2], V[5]=B[1,2]
    # C[0,2]=W[4], C[1,2]=W[5]
    
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    U[0][col] = 1; V[2][col] = 1; W[4][col] = 1; col += 1
    U[1][col] = 1; V[5][col] = 1; W[4][col] = 1; col += 1
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    U[3][col] = 1; V[2][col] = 1; W[5][col] = 1; col += 1
    U[4][col] = 1; V[5][col] = 1; W[5][col] = 1; col += 1
    
    # Part 3: A2*B2 (2x1 * 1x3), schoolbook rank 6
    # A2: U[2]=A[0,2], U[5]=A[1,2]
    # B2: V[6]=B[2,0], V[7]=B[2,1], V[8]=B[2,2]
    # C[0,0]=W[0], C[0,1]=W[1], C[0,2]=W[4]
    # C[1,0]=W[2], C[1,1]=W[3], C[1,2]=W[5]
    
    for k in range(3):
        # A[0,2] * B[2,k] -> C[0,k] = W[k*2+0]
        U[2][col] = 1; V[6+k][col] = 1; W[k*2+0][col] = 1; col += 1
        # A[1,2] * B[2,k] -> C[1,k] = W[k*2+1]
        U[5][col] = 1; V[6+k][col] = 1; W[k*2+1][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
