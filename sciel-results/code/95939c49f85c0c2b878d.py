"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    
    # Strassen 2x2x2 decomposition (rank 7)
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    U1[0][0]=1; U1[3][0]=1
    V1[0][0]=1; V1[3][0]=1
    W1[0][0]=1; W1[2][0]=-1; W1[1][0]=-1; W1[3][0]=1
    
    U1[2][1]=1; U1[3][1]=1
    V1[0][1]=1
    W1[0][1]=1; W1[1][1]=-1
    
    U1[0][2]=1
    V1[1][2]=1; V1[3][2]=-1
    W1[2][2]=1; W1[3][2]=1
    
    U1[3][3]=1
    V1[2][3]=1; V1[0][3]=-1
    W1[0][3]=1; W1[1][3]=1
    
    U1[0][4]=1; U1[1][4]=1
    V1[3][4]=1
    W1[0][4]=-1; W1[3][4]=1
    
    U1[2][5]=1; U1[0][5]=-1
    V1[0][5]=1; V1[1][5]=1
    W1[3][5]=1
    
    U1[1][6]=1; U1[3][6]=-1
    V1[2][6]=1; V1[3][6]=1
    W1[0][6]=1
    
    # 1x2x3 trivial decomposition (rank 6)
    U2 = [[0]*6 for _ in range(2)]
    V2 = [[0]*6 for _ in range(3)]
    W2 = [[0]*6]
    for s2 in range(6):
        b = s2 // 3
        d = s2 % 3
        U2[b][s2] = 1
        V2[d][s2] = 1
        W2[0][s2] = 1
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for s1 in range(7):
        for s2 in range(6):
            s = s1*6 + s2
            w2_val = W2[0][s2]
            if w2_val == 0:
                continue
            for i in range(2):
                for a in range(2):
                    u1_val = U1[i*2+a][s1]
                    if u1_val == 0:
                        continue
                    for b in range(2):
                        u2_val = U2[b][s2]
                        if u2_val == 0:
                            continue
                        j = 2*a + b
                        U[i*4 + j][s] = u1_val * u2_val
            for a in range(2):
                for c in range(2):
                    v1_val = V1[a*2+c][s1]
                    if v1_val == 0:
                        continue
                    for d in range(3):
                        v2_val = V2[d][s2]
                        if v2_val == 0:
                            continue
                        k = 3*c + d
                        for b in range(2):
                            j = 2*a + b
                            V[j*6 + k][s] = v1_val * v2_val
            for c in range(2):
                for i in range(2):
                    w1_val = W1[c*2+i][s1]
                    if w1_val == 0:
                        continue
                    for d in range(3):
                        k = 3*c + d
                        W[k*2 + i][s] = w1_val * w2_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
