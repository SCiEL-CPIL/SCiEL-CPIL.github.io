def construct():
    # Strassen decomposition of M_{2,2,2}
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    U1[0][0]=1; U1[3][0]=1; V1[0][0]=1; V1[3][0]=1; W1[0][0]=1; W1[3][0]=1
    U1[2][1]=1; U1[3][1]=1; V1[0][1]=1; W1[2][1]=1
    U1[0][2]=1; V1[1][2]=1; V1[3][2]=-1; W1[1][2]=1
    U1[3][3]=1; V1[2][3]=1; V1[0][3]=-1; W1[0][3]=1; W1[2][3]=1
    U1[0][4]=1; U1[1][4]=1; V1[3][4]=1; W1[0][4]=-1; W1[1][4]=1
    U1[2][5]=1; U1[0][5]=-1; V1[0][5]=1; V1[1][5]=1; W1[3][5]=1
    U1[1][6]=1; U1[3][6]=-1; V1[2][6]=1; V1[3][6]=1; W1[0][6]=1; W1[3][6]=1
    
    n, m, p = 4, 4, 5
    rank1 = 49
    rank2 = 16
    rank = rank1 + rank2
    
    U = [[0]*rank for _ in range(n*m)]  # 16
    V = [[0]*rank for _ in range(m*p)]  # 20
    W = [[0]*rank for _ in range(p*n)]  # 20
    
    # Kronecker part: M_{4,4,4} embedded in first 4 cols of B, first 4 cols of C
    for r1 in range(7):
        for r2 in range(7):
            r = r1*7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1*2 + i2
                            j = j1*2 + j2
                            u_idx = i*m + j  # i*4+j
                            U[u_idx][r] += U1[i1*2+j1][r1] * U1[i2*2+j2][r2]
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1*2 + j2
                            k = k1*2 + k2
                            v_idx = j*p + k  # j*5+k
                            V[v_idx][r] += V1[j1*2+k1][r1] * V1[j2*2+k2][r2]
            
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1*2 + k2
                            i = i1*2 + i2
                            w_idx = k*n + i  # k*4+i
                            W[w_idx][r] += W1[k1*2+i1][r1] * W1[k2*2+i2][r2]
    
    # Schoolbook for 5th column
    col = 0
    for i in range(n):
        for j in range(m):
            U[i*m+j][rank1+col] = 1
            V[j*p+4][rank1+col] = 1  # 5th column of B (k=4)
            W[4*n+i][rank1+col] = 1  # 5th column of C (k=4)
            col += 1
    
    return {"U": U, "V": V, "W": W}
