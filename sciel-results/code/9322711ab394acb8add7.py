"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U1: 4x7, V1: 4x7, W1: 4x7
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # M2 = (c+d)e
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[2][1] = 1; W1[3][1] = -1
    
    # M3 = a(f-h)
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[1][2] = 1; W1[3][2] = 1
    
    # M4 = d(g-c)
    U1[3][3] = 1; U1[2][3] = -1
    V1[2][3] = 1
    W1[2][3] = 1; W1[0][3] = 1
    
    # M5 = (a+b)h
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[1][4] = 1; W1[0][4] = -1
    
    # M6 = (c-a)(e+f)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    # M7 = (b-d)(g+h)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1; W1[3][6] = -1
    
    # Schoolbook decomposition for 1x2x4
    # U2: 2x8, V2: 8x8, W2: 4x8
    U2 = [[0]*8 for _ in range(2)]
    V2 = [[0]*8 for _ in range(8)]
    W2 = [[0]*8 for _ in range(4)]
    
    # A is 1x2 (indices 0,1), B is 2x4 (indices 0..7), C is 1x4 (indices 0..3)
    # Terms:
    # j=0: A[0] * B[0..3] -> C[0..3]
    # j=1: A[1] * B[4..7] -> C[0..3]
    
    for k in range(4):
        # j=0, k=k
        U2[0][k] = 1
        V2[k][k] = 1
        W2[k][k] = 1
        
        # j=1, k=k
        U2[1][4+k] = 1
        V2[4+k][4+k] = 1
        W2[k][4+k] = 1
        
    # Construct total U, V, W
    # Rank = 7 * 8 = 56
    rank = 56
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Mapping indices
    # U: idx_A = r1*4 + c1*2 + c2. i1 = r1*2 + c1. i2 = c2.
    # V: idx_B = (r1*2 + r2)*8 + c1*4 + c2. j1 = r1*2 + c1. j2 = r2*4 + c2.
    # W: idx_C = r1*8 + c1*4 + c2. k1 = r1*2 + c1. k2 = c2.
    
    for c1 in range(7):
        for c2 in range(8):
            col = c1 * 8 + c2
            
            # U
            for r1 in range(2):
                for c1_a in range(2):
                    for c2_a in range(2):
                        i1 = r1 * 2 + c1_a
                        i2 = c2_a
                        idx_A = r1 * 4 + c1_a * 2 + c2_a
                        val = U1[i1][c1] * U2[i2][c2]
                        if val != 0:
                            U[idx_A][col] += val
                            
            # V
            for r1_b in range(2):
                for c1_b in range(2):
                    for r2_b in range(2):
                        for c2_b in range(4):
                            j1 = r1_b * 2 + c1_b
                            j2 = r2_b * 4 + c2_b
                            idx_B = (r1_b * 2 + r2_b) * 8 + c1_b * 4 + c2_b
                            val = V1[j1][c1] * V2[j2][c2]
                            if val != 0:
                                V[idx_B][col] += val
                                
            # W
            for r1_c in range(2):
                for c1_c in range(2):
                    for c2_c in range(4):
                        k1 = r1_c * 2 + c1_c
                        k2 = c2_c
                        idx_C = r1_c * 8 + c1_c * 4 + c2_c
                        val = W1[k1][c1] * W2[k2][c2]
                        if val != 0:
                            W[idx_C][col] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
