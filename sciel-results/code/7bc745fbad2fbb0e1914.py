import time

def construct():
    n, m, p = 2, 2, 4
    nm, mp, pn = n * m, m * p, p * n
    
    def verify(U, V, W):
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def get_rank_14():
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B)
        # M1: (a0+a3)(b0+b3) -> c0, c3
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[3][0]=1
        W[0][0]=1; W[3][0]=1
        
        # M2: (a2+a3)b0 -> c0(-1), c2(1)
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[0][1]=-1; W[2][1]=1
        
        # M3: a0(b2-b3) -> c1(1), c3(1)
        U[0][2]=1
        V[2][2]=1; V[3][2]=-1
        W[1][2]=1; W[3][2]=1
        
        # M4: a3(b1-b0) -> c0(1), c2(1)
        U[3][3]=1
        V[1][3]=1; V[0][3]=-1
        W[0][3]=1; W[2][3]=1
        
        # M5: (a0+a1)b3 -> c0(-1), c1(1)
        U[0][4]=1; U[1][4]=1
        V[3][4]=1
        W[0][4]=-1; W[1][4]=1
        
        # M6: (a2-a0)(b0+b1) -> c3(1)
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        W[3][5]=1
        
        # M7: (a1-a3)(b2+b3) -> c0(1)
        U[1][6]=1; U[3][6]=-1
        V[2][6]=1; V[3][6]=1
        W[0][6]=1
        
        # Right block (cols 2,3 of B)
        # N1: (a0+a3)(b4+b7) -> c4, c7
        U[0][7]=1; U[3][7]=1
        V[4][7]=1; V[7][7]=1
        W[4][7]=1; W[7][7]=1
        
        # N2: (a2+a3)b4 -> c4(-1), c6(1)
        U[2][8]=1; U[3][8]=1
        V[4][8]=1
        W[4][8]=-1; W[6][8]=1
        
        # N3: a0(b6-b7) -> c5(1), c7(1)
        U[0][9]=1
        V[6][9]=1; V[7][9]=-1
        W[5][9]=1; W[7][9]=1
        
        # N4: a3(b5-b4) -> c4(1), c6(1)
        U[3][10]=1
        V[5][10]=1; V[4][10]=-1
        W[4][10]=1; W[6][10]=1
        
        # N5: (a0+a1)b7 -> c4(-1), c5(1)
        U[0][11]=1; U[1][11]=1
        V[7][11]=1
        W[4][11]=-1; W[5][11]=1
        
        # N6: (a2-a0)(b4+b5) -> c7(1)
        U[2][12]=1; U[0][12]=-1
        V[4][12]=1; V[5][12]=1
        W[7][12]=1
        
        # N7: (a1-a3)(b6+b7) -> c4(1)
        U[1][13]=1; U[3][13]=-1
        V[6][13]=1; V[7][13]=1
        W[4][13]=1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None

    def get_naive():
        rank = 16
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(8)]
        W = [[0]*rank for _ in range(8)]
        r = 0
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    U[i*2+j][r] = 1
                    V[j*4+k][r] = 1
                    W[k*2+i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    res = get_rank_14()
    if res:
        return res
    return get_naive()
