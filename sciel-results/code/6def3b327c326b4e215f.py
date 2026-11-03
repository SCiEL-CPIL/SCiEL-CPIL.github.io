DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 factors (rank 7)
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # c=0: (a00+a11)(b00+b11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    # c=1: (a10+a11)b00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    # c=2: a00(b01-b11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    # c=3: a11(b10-b00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[2][3] = 1; W2[0][3] = 1
    # c=4: (a00+a01)b11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[1][4] = -1; W2[3][4] = 1
    # c=5: (a10-a00)(b00+b01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[0][5] = 1; W2[1][5] = 1
    # c=6: (a01-a11)(b10+b11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = -1; W2[2][6] = 1
    
    # Schoolbook 3x3 factors (rank 27)
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][col] = 1
                V3[j*3+k][col] = 1
                W3[k*3+i][col] = 1
                col += 1
    
    # Kronecker product: rank = 27 * 7 = 189
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for c1 in range(27):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(3):
                for i2 in range(2):
                    for j1 in range(3):
                        for j2 in range(2):
                            u_val = U3[i1*3+j1][c1] * U2[i2*2+j2][c2]
                            if u_val != 0:
                                row_u = (i1*2+i2)*6 + (j1*2+j2)
                                U[row_u][c] = u_val
            for j1 in range(3):
                for j2 in range(2):
                    for k1 in range(3):
                        for k2 in range(2):
                            v_val = V3[j1*3+k1][c1] * V2[j2*2+k2][c2]
                            if v_val != 0:
                                row_v = (j1*2+j2)*6 + (k1*2+k2)
                                V[row_v][c] = v_val
            for k1 in range(3):
                for k2 in range(2):
                    for i1 in range(3):
                        for i2 in range(2):
                            w_val = W3[k1*3+i1][c1] * W2[k2*2+i2][c2]
                            if w_val != 0:
                                row_w = (k1*2+k2)*6 + (i1*2+i2)
                                W[row_w][c] = w_val
    
    return {"U": U, "V": V, "W": W}
