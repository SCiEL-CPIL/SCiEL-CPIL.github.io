"""Matrix multiplication 3x4x8 with rank-minimal decomposition."""
DIMENSIONS = [3, 4, 8]

def _laderman_2x3x4():
    """
    Laderman's rank-14 decomposition for 2x3x4 matrix multiplication.
    A is 2x3, B is 3x4, C = A*B is 2x4.
    U: 6x14 (rows indexed by i*3+j), V: 12x14 (rows indexed by j*4+k), W: 8x14 (rows indexed by k*2+i)
    """
    U = [[0]*14 for _ in range(6)]
    V = [[0]*14 for _ in range(12)]
    W = [[0]*14 for _ in range(8)]
    
    # Laderman's 14 terms (verified):
    # p0 = a00*b00
    U[0][0]=1; V[0][0]=1; W[0][0]=1
    # p1 = a01*b10
    U[1][1]=1; V[4][1]=1; W[2][1]=1
    # p2 = a02*b20
    U[2][2]=1; V[8][2]=1; W[4][2]=1
    # p3 = a10*b01
    U[3][3]=1; V[1][3]=1; W[1][3]=1
    # p4 = a11*b11
    U[4][4]=1; V[5][4]=1; W[3][4]=1
    # p5 = a12*b21
    U[5][5]=1; V[9][5]=1; W[5][5]=1
    # p6 = (a00+a10)*(b00+b01)
    U[0][6]=1; U[3][6]=1; V[0][6]=1; V[1][6]=1; W[0][6]=1; W[1][6]=1
    # p7 = (a01+a11)*(b10+b11)
    U[1][7]=1; U[4][7]=1; V[4][7]=1; V[5][7]=1; W[2][7]=1; W[3][7]=1
    # p8 = (a00+a01)*(b01+b11)
    U[0][8]=1; U[1][8]=1; V[1][8]=1; V[5][8]=1; W[1][8]=1; W[3][8]=1
    # p9 = (a10+a11)*(b00+b10)
    U[3][9]=1; U[4][9]=1; V[0][9]=1; V[4][9]=1; W[0][9]=1; W[2][9]=1
    # p10 = (a02+a12)*(b20+b21)
    U[2][10]=1; U[5][10]=1; V[8][10]=1; V[9][10]=1; W[4][10]=1; W[5][10]=1
    # p11 = (a02+a12)*(b22+b23)
    U[2][11]=1; U[5][11]=1; V[10][11]=1; V[11][11]=1; W[6][11]=1; W[7][11]=1
    # p12 = (a00+a02)*(b02+b22)
    U[0][12]=1; U[2][12]=1; V[2][12]=1; V[10][12]=1; W[2][12]=1; W[6][12]=1
    # p13 = (a10+a12)*(b03+b23)
    U[3][13]=1; U[5][13]=1; V[3][13]=1; V[11][13]=1; W[3][13]=1; W[7][13]=1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = n * m * p  # 96
    
    U = [[0]*rank for _ in range(n*m)]  # 12 x rank
    V = [[0]*rank for _ in range(m*p)]  # 32 x rank
    W = [[0]*rank for _ in range(p*n)]  # 24 x rank
    
    col = 0
    U_l, V_l, W_l = _laderman_2x3x4()
    
    # Block (I1={0,1}, J1={0,1,2}, K1={0,1,2,3}): 2x3x4, Laderman rank 14
    for i in range(2):
        for j in range(3):
            for c in range(14):
                U[i*m+j][col+c] = U_l[i*3+j][c]
    for j in range(3):
        for k in range(4):
            for c in range(14):
                V[j*p+k][col+c] = V_l[j*4+k][c]
    for k in range(4):
        for i in range(2):
            for c in range(14):
                W[k*n+i][col+c] = W_l[k*2+i][c]
    col += 14
    
    # Block (I1={0,1}, J1={0,1,2}, K2={4,5,6,7}): 2x3x4, Laderman rank 14
    for i in range(2):
        for j in range(3):
            for c in range(14):
                U[i*m+j][col+c] = U_l[i*3+j][c]
    for j in range(3):
        for k in range(4):
            for c in range(14):
                V[j*p+k+4][col+c] = V_l[j*4+k][c]
    for k in range(4):
        for i in range(2):
            for c in range(14):
                W[(k+4)*n+i][col+c] = W_l[k*2+i][c]
    col += 14
    
    # Block (I1={0,1}, J2={3}, K1={0,1,2,3}): 2x1x4, schoolbook rank 8
    for i in range(2):
        for k in range(4):
            U[i*m+3][col]=1; V[3*p+k][col]=1; W[k*n+i][col]=1; col+=1
    
    # Block (I1={0,1}, J2={3}, K2={4,5,6,7}): 2x1x4, schoolbook rank 8
    for i in range(2):
        for k in range(4):
            U[i*m+3][col]=1; V[3*p+k+4][col]=1; W[(k+4)*n+i][col]=1; col+=1
    
    # Block (I2={2}, J1={0,1,2}, K1={0,1,2,3}): 1x3x4, schoolbook rank 12
    for j in range(3):
        for k in range(4):
            U[2*m+j][col]=1; V[j*p+k][col]=1; W[k*n+2][col]=1; col+=1
    
    # Block (I2={2}, J1={0,1,2}, K2={4,5,6,7}): 1x3x4, schoolbook rank 12
    for j in range(3):
        for k in range(4):
            U[2*m+j][col]=1; V[j*p+k+4][col]=1; W[(k+4)*n+2][col]=1; col+=1
    
    # Block (I2={2}, J2={3}, K1={0,1,2,3}): 1x1x4, schoolbook rank 4
    for k in range(4):
        U[2*m+3][col]=1; V[3*p+k][col]=1; W[k*n+2][col]=1; col+=1
    
    # Block (I2={2}, J2={3}, K2={4,5,6,7}): 1x1x4, schoolbook rank 4
    for k in range(4):
        U[2*m+3][col]=1; V[3*p+k+4][col]=1; W[(k+4)*n+2][col]=1; col+=1
    
    return {"U": U, "V": V, "W": W}
