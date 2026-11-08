"""Rewritten program using Strassen's algorithm for rank reduction."""

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen 2x2 factors
    # U2, V2, W2 are 4x7 matrices
    # Indices: U2[i*2+j], V2[j*2+k], W2[k*2+i]
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Col 0: P0 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Col 1: P1 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # Col 2: P2 = A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # Col 3: P3 = A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # Col 4: P4 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # Col 5: P5 = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Col 6: P6 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    # Construct 4x4x4 factors (Rank 49)
    # U444[16][49], V444[16][49], W444[16][49]
    U444 = [[0]*49 for _ in range(16)]
    V444 = [[0]*49 for _ in range(16)]
    W444 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2*i1 + i0
                    j = 2*j1 + j0
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            U444[i*4+j][r] = U2[i1*2+j1][r1] * U2[i0*2+j0][r2]
    
    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    j = 2*j1 + j0
                    k = 2*k1 + k0
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            V444[j*4+k][r] = V2[j1*2+k1][r1] * V2[j0*2+k0][r2]
                            
    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    k = 2*k1 + k0
                    i = 2*i1 + i0
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            W444[k*4+i][r] = W2[k1*2+i1][r1] * W2[k0*2+i0][r2]

    # Construct 4x4x8 factors (Rank 98)
    # U is same for both halves
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U[i*4+j][r] = U444[i*4+j][r]
                U[i*4+j][r+49] = U444[i*4+j][r]
                
    # V: first half uses k in 0..3, second half uses k in 4..7
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j*8+k][r] = V444[j*4+k][r]
                V[j*8+(k+4)][r+49] = V444[j*4+k][r]
                
    # W: first half uses k in 0..3, second half uses k in 4..7
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[k*4+i][r] = W444[k*4+i][r]
                W[(k+4)*4+i][r+49] = W444[k*4+i][r]

    return {"U": U, "V": V, "W": W}
