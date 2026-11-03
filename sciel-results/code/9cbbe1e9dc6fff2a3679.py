def construct():
    n, m, p = 3, 4, 8
    total_rank = 8 + 8 + 14 + 14  # 44
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    
    # Block 1: 1x2x4 schoolbook at i0=0, j0=0, k0=0
    # U rows: i=0, j=0,1 -> global row 0*4+0=0, 0*4+1=1
    # V rows: j=0,1; k=0,1,2,3 -> global row 0*8+0=0, 0*8+1=1, 0*8+2=2, 0*8+3=3, 1*8+0=8, 1*8+1=9, 1*8+2=10, 1*8+3=11
    # W rows: k=0,1,2,3; i=0 -> global row 0*3+0=0, 1*3+0=3, 2*3+0=6, 3*3+0=9
    for j in range(2):
        for k in range(4):
            U[0*4+j][col] = 1
            V[j*8+k][col] = 1
            W[k*3+0][col] = 1
            col += 1
    
    # Block 2: 1x2x4 schoolbook at i0=0, j0=2, k0=4
    # U rows: i=0, j=0,1 -> global row 0*4+2=2, 0*4+3=3
    # V rows: j=0,1; k=0,1,2,3 -> global row (2+0)*8+(4+0)=20, 21, 22, 23, (2+1)*8+(4+0)=28, 29, 30, 31
    # W rows: k=0,1,2,3; i=0 -> global row (4+0)*3+0=12, (4+1)*3+0=15, (4+2)*3+0=18, (4+3)*3+0=21
    for j in range(2):
        for k in range(4):
            U[0*4+(2+j)][col] = 1
            V[(2+j)*8+(4+k)][col] = 1
            W[(4+k)*3+0][col] = 1
            col += 1
    
    # Block 3: 2x2x2 Strassen at i0=1, j0=0, k0=0
    # U rows: i=0,1; j=0,1 -> global row (1+0)*4+(0+0)=4, (1+0)*4+(0+1)=5, (1+1)*4+(0+0)=8, (1+1)*4+(0+1)=9
    # V rows: j=0,1; k=0,1 -> global row (0+0)*8+(0+0)=0, (0+0)*8+(0+1)=1, (0+1)*8+(0+0)=8, (0+1)*8+(0+1)=9
    # W rows: k=0,1; i=0,1 -> global row (0+0)*3+(1+0)=1, (0+0)*3+(1+1)=2, (0+1)*3+(1+0)=4, (0+1)*3+(1+1)=5
    
    # Strassen M1 = (a+d)(e+h)
    U[4][col] = 1; U[9][col] = 1
    V[0][col] = 1; V[9][col] = 1
    W[1][col] = 1; W[5][col] = 1
    col += 1
    
    # M2 = (c+d)e
    U[8][col] = 1; U[9][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    # M3 = a(f-h)
    U[4][col] = 1
    V[1][col] = 1; V[9][col] = -1
    W[4][col] = 1
    col += 1
    
    # M4 = d(g-e)
    U[9][col] = 1
    V[8][col] = 1; V[0][col] = -1
    W[2][col] = 1
    col += 1
    
    # M5 = (a+b)h
    U[4][col] = 1; U[5][col] = 1
    V[9][col] = 1
    W[5][col] = 1
    col += 1
    
    # M6 = (c-d)(f+h)
    U[8][col] = 1; U[9][col] = -1
    V[1][col] = 1; V[9][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (b-d)(e+g)
    U[5][col] = 1; U[9][col] = -1
    V[0][col] = 1; V[8][col] = 1
    W[1][col] = 1; W[5][col] = 1
    col += 1
    
    # Block 4: 2x2x2 Strassen at i0=1, j0=0, k0=2
    # U rows: same as block 3: 4,5,8,9
    # V rows: j=0,1; k=0,1 -> global row (0+0)*8+(2+0)=2, (0+0)*8+(2+1)=3, (0+1)*8+(2+0)=10, (0+1)*8+(2+1)=11
    # W rows: k=0,1; i=0,1 -> global row (2+0)*3+(1+0)=7, (2+0)*3+(1+1)=8, (2+1)*3+(1+0)=10, (2+1)*3+(1+1)=11
    
    # M1
    U[4][col] = 1; U[9][col] = 1
    V[2][col] = 1; V[11][col] = 1
    W[7][col] = 1; W[11][col] = 1
    col += 1
    
    # M2
    U[8][col] = 1; U[9][col] = 1
    V[2][col] = 1
    W[8][col] = 1
    col += 1
    
    # M3
    U[4][col] = 1
    V[3][col] = 1; V[11][col] = -1
    W[10][col] = 1
    col += 1
    
    # M4
    U[9][col] = 1
    V[10][col] = 1; V[2][col] = -1
    W[8][col] = 1
    col += 1
    
    # M5
    U[4][col] = 1; U[5][col] = 1
    V[11][col] = 1
    W[11][col] = 1
    col += 1
    
    # M6
    U[8][col] = 1; U[9][col] = -1
    V[3][col] = 1; V[11][col] = 1
    W[10][col] = 1
    col += 1
    
    # M7
    U[5][col] = 1; U[9][col] = -1
    V[2][col] = 1; V[10][col] = 1
    W[7][col] = 1; W[11][col] = 1
    col += 1
    
    # Block 5: 2x2x2 Strassen at i0=1, j0=2, k0=4
    # U rows: i=0,1; j=0,1 -> global row (1+0)*4+(2+0)=6, (1+0)*4+(2+1)=7, (1+1)*4+(2+0)=10, (1+1)*4+(2+1)=11
    # V rows: j=0,1; k=0,1 -> global row (2+0)*8+(4+0)=20, (2+0)*8+(4+1)=21, (2+1)*8+(4+0)=28, (2+1)*8+(4+1)=29
    # W rows: k=0,1; i=0,1 -> global row (4+0)*3+(1+0)=13, (4+0)*3+(1+1)=14, (4+1)*3+(1+0)=16, (4+1)*3+(1+1)=17
    
    # M1
    U[6][col] = 1; U[11][col] = 1
    V[20][col] = 1; V[29][col] = 1
    W[13][col] = 1; W[17][col] = 1
    col += 1
    
    # M2
    U[10][col] = 1; U[11][col] = 1
    V[20][col] = 1
    W[14][col] = 1
    col += 1
    
    # M3
    U[6][col] = 1
    V[21][col] = 1; V[29][col] = -1
    W[16][col] = 1
    col += 1
    
    # M4
    U[11][col] = 1
    V[28][col] = 1; V[20][col] = -1
    W[14][col] = 1
    col += 1
    
    # M5
    U[6][col] = 1; U[7][col] = 1
    V[29][col] = 1
    W[17][col] = 1
    col += 1
    
    # M6
    U[10][col] = 1; U[11][col] = -1
    V[21][col] = 1; V[29][col] = 1
    W[16][col] = 1
    col += 1
    
    # M7
    U[7][col] = 1; U[11][col] = -1
    V[20][col] = 1; V[28][col] = 1
    W[13][col] = 1; W[17][col] = 1
    col += 1
    
    # Block 6: 2x2x2 Strassen at i0=1, j0=2, k0=6
    # U rows: same as block 5: 6,7,10,11
    # V rows: j=0,1; k=0,1 -> global row (2+0)*8+(6+0)=22, (2+0)*8+(6+1)=23, (2+1)*8+(6+0)=30, (2+1)*8+(6+1)=31
    # W rows: k=0,1; i=0,1 -> global row (6+0)*3+(1+0)=19, (6+0)*3+(1+1)=20, (6+1)*3+(1+0)=22, (6+1)*3+(1+1)=23
    
    # M1
    U[6][col] = 1; U[11][col] = 1
    V[22][col] = 1; V[31][col] = 1
    W[19][col] = 1; W[23][col] = 1
    col += 1
    
    # M2
    U[10][col] = 1; U[11][col] = 1
    V[22][col] = 1
    W[20][col] = 1
    col += 1
    
    # M3
    U[6][col] = 1
    V[23][col] = 1; V[31][col] = -1
    W[22][col] = 1
    col += 1
    
    # M4
    U[11][col] = 1
    V[30][col] = 1; V[22][col] = -1
    W[20][col] = 1
    col += 1
    
    # M5
    U[6][col] = 1; U[7][col] = 1
    V[31][col] = 1
    W[23][col] = 1
    col += 1
    
    # M6
    U[10][col] = 1; U[11][col] = -1
    V[23][col] = 1; V[31][col] = 1
    W[22][col] = 1
    col += 1
    
    # M7
    U[7][col] = 1; U[11][col] = -1
    V[22][col] = 1; V[30][col] = 1
    W[19][col] = 1; W[23][col] = 1
    col += 1
    
    assert col == total_rank, f"col={col}, total_rank={total_rank}"
    
    return {"U": U, "V": V, "W": W}
