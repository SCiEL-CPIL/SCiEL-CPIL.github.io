"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Tensor product: [6,6,6] = [2,2,2] ⊗ [3,3,3]
    # Strassen for 2x2 (rank 7), schoolbook for 3x3 (rank 27)
    # Total rank = 7 * 27 = 189
    
    rank1 = 7
    rank2 = 27
    rank = rank1 * rank2
    
    # Strassen 2x2x2
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U1[0][0]=1; U1[3][0]=1; V1[0][0]=1; V1[3][0]=1
    W1[0][0]=1; W1[1][0]=1; W1[2][0]=1; W1[3][0]=-1
    # M2 = (A10+A11)B00
    U1[2][1]=1; U1[3][1]=1; V1[0][1]=1
    W1[2][1]=1; W1[3][1]=-1
    # M3 = A00(B10-B11)
    U1[0][2]=1; V1[2][2]=1; V1[3][2]=-1
    W1[1][2]=1; W1[3][2]=1
    # M4 = A11(B01-B10)
    U1[3][3]=1; V1[1][3]=1; V1[2][3]=-1
    W1[0][3]=1; W1[2][3]=1
    # M5 = (A00+A01)B11
    U1[0][4]=1; U1[1][4]=1; V1[3][4]=1
    W1[0][4]=-1; W1[1][4]=1
    # M6 = (A10-A00)(B00+B10)
    U1[2][5]=1; U1[0][5]=-1; V1[0][5]=1; V1[2][5]=1
    W1[3][5]=1
    # M7 = (A01-A11)(B01+B11)
    U1[1][6]=1; U1[3][6]=-1; V1[1][6]=1; V1[3][6]=1
    W1[0][6]=1; W1[1][6]=1
    
    # Schoolbook 3x3x3
    U2 = [[0]*27 for _ in range(9)]
    V2 = [[0]*27 for _ in range(9)]
    W2 = [[0]*27 for _ in range(9)]
    for i2 in range(3):
        for j2 in range(3):
            for k2 in range(3):
                c2 = i2*9 + j2*3 + k2
                U2[i2*3+j2][c2] = 1
                V2[j2*3+k2][c2] = 1
                W2[k2*3+i2][c2] = 1
    
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for i2 in range(3):
                    for j2 in range(3):
                        for k2 in range(3):
                            i = i1*3 + i2
                            j = j1*3 + j2
                            k = k1*3 + k2
                            for c1 in range(rank1):
                                for c2 in range(rank2):
                                    c = c1*rank2 + c2
                                    u_val = U1[i1*2+j1][c1] * U2[i2*3+j2][c2]
                                    v_val = V1[j1*2+k1][c1] * V2[j2*3+k2][c2]
                                    w_val = W1[k1*2+i1][c1] * W2[k2*3+i2][c2]
                                    if u_val != 0:
                                        U[i*6+j][c] = u_val
                                    if v_val != 0:
                                        V[j*6+k][c] = v_val
                                    if w_val != 0:
                                        W[k*6+i][c] = w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
