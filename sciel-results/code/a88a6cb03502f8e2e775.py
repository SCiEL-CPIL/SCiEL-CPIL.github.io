"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Kronecker product: 4x4x6 = (2x2x2) ⊗ (2x2x3)
    # Factor 1: 2x2x2 matrix multiplication, rank 7 (Strassen)
    # Factor 2: 2x2x3 matrix multiplication, rank 11 (Strassen for 2 cols + trivial for 3rd)
    # Total rank = 7 * 11 = 77
    
    # Strassen decomposition for 2x2x2:
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)(b11+b22)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # M2 = (a21+a22)b11
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1; W1[3][1] = -1
    
    # M3 = a11(b12-b22)
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1
    
    # M4 = a22(b21-b11)
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1
    
    # M5 = (a11+a12)b22
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1
    
    # For 2x2x3: A is 2x2, B is 2x3, C is 2x3
    # Use Strassen for first 2 columns (rank 7) + trivial for 3rd column (rank 4)
    # Total rank for 2x2x3 = 7 + 4 = 11
    
    U2 = [[0]*11 for _ in range(4)]
    V2 = [[0]*11 for _ in range(6)]
    W2 = [[0]*11 for _ in range(6)]
    
    # Strassen for first 2 columns (mapping 2x2x2 to 2x2x3 with cols 0,1)
    for c in range(7):
        U2[0][c] = U1[0][c]
        U2[1][c] = U1[1][c]
        U2[2][c] = U1[2][c]
        U2[3][c] = U1[3][c]
        
        # V2 rows: j*3+k where j in 0..1, k in 0..2
        # For 2x2x2, V1 rows: j*2+k where j in 0..1, k in 0..1
        # Map V1[j*2+k] -> V2[j*3+k] for k=0,1; V2[j*3+2] = 0
        V2[0][c] = V1[0][c]  # j=0,k=0
        V2[1][c] = V1[1][c]  # j=0,k=1
        V2[2][c] = 0         # j=0,k=2
        V2[3][c] = V1[2][c]  # j=1,k=0
        V2[4][c] = V1[3][c]  # j=1,k=1
        V2[5][c] = 0         # j=1,k=2
        
        # W2 rows: k*2+i where k in 0..2, i in 0..1
        # For 2x2x2, W1 rows: k*2+i where k in 0..1, i in 0..1
        # Map W1[k*2+i] -> W2[k*2+i] for k=0,1; W2[4+i] = 0 for i=0,1
        W2[0][c] = W1[0][c]  # k=0,i=0
        W2[1][c] = W1[1][c]  # k=0,i=1
        W2[2][c] = W1[2][c]  # k=1,i=0
        W2[3][c] = W1[3][c]  # k=1,i=1
        W2[4][c] = 0         # k=2,i=0
        W2[5][c] = 0         # k=2,i=1
    
    # Trivial terms for 3rd column (col index 2 in B, col index 2 in C)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # Term 7: A[0,0]*B[0,2] -> C[0,2]
    U2[0][7] = 1
    V2[2][7] = 1
    W2[4][7] = 1
    
    # Term 8: A[0,1]*B[1,2] -> C[0,2]
    U2[1][8] = 1
    V2[5][8] = 1
    W2[4][8] = 1
    
    # Term 9: A[1,0]*B[0,2] -> C[1,2]
    U2[2][9] = 1
    V2[2][9] = 1
    W2[5][9] = 1
    
    # Term 10: A[1,1]*B[1,2] -> C[1,2]
    U2[3][10] = 1
    V2[5][10] = 1
    W2[5][10] = 1
    
    # Now construct the full 4x4x6 decomposition via Kronecker product
    rank = 7 * 11
    U = [[0]*rank for _ in range(n*m)]  # (16, 77)
    V = [[0]*rank for _ in range(m*p)]  # (24, 77)
    W = [[0]*rank for _ in range(p*n)]  # (24, 77)
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1][c1]
                            u2 = U2[i2*2+j2][c2]
                            if u1 != 0 and u2 != 0:
                                # Full U row: (i1*2+i2)*4 + (j1*2+j2)
                                row = (i1*2+i2)*4 + (j1*2+j2)
                                U[row][c] = u1 * u2
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2+k1][c1]
                            v2 = V2[j2*3+k2][c2]
                            if v1 != 0 and v2 != 0:
                                # Full V row: (j1*2+j2)*6 + (k1*3+k2)
                                row = (j1*2+j2)*6 + (k1*3+k2)
                                V[row][c] = v1 * v2
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1][c1]
                            w2 = W2[k2*2+i2][c2]
                            if w1 != 0 and w2 != 0:
                                # Full W row: (k1*2+k2)*4 + (i1*2+i2)
                                row = (k1*2+k2)*4 + (i1*2+i2)
                                W[row][c] = w1 * w2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
