DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    def strassen_222():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # m1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # m2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        # m3
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        # m4
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        # m5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[1][4] = 1; W[0][4] = -1
        # m6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # m7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[3][6] = 1; W[0][6] = 1
        return U, V, W
    
    total_rank = 62
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    # [2,2,2] Strassen: i=0,1; j=0,1; k=0,1
    U222, V222, W222 = strassen_222()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m+j][col+c] += U222[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V[j*p+k][col+c] += V222[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W[k*n+i][col+c] += W222[k*2+i][c]
    col += 7
    
    # Schoolbook helper
    def schoolbook_block(i0, j0, k0, sn, sm, sp):
        nonlocal col
        for i in range(sn):
            for j in range(sm):
                for k in range(sp):
                    U[(i0+i)*m + (j0+j)][col] = 1
                    V[(j0+j)*p + (k0+k)][col] = 1
                    W[(k0+k)*n + (i0+i)][col] = 1
                    col += 1
    
    # [2,2,1]: i=0,1; j=0,1; k=2
    schoolbook_block(0, 0, 2, 2, 2, 1)
    # [2,2,4]: i=0,1; j=0,1; k=3..6
    schoolbook_block(0, 0, 3, 2, 2, 4)
    # [1,2,3]: i=2; j=0,1; k=0..2
    schoolbook_block(2, 0, 0, 1, 2, 3)
    # [1,2,4]: i=2; j=0,1; k=3..6
    schoolbook_block(2, 0, 3, 1, 2, 4)
    # [2,1,3]: i=0,1; j=2; k=0..2
    schoolbook_block(0, 2, 0, 2, 1, 3)
    # [2,1,4]: i=0,1; j=2; k=3..6
    schoolbook_block(0, 2, 3, 2, 1, 4)
    # [1,1,3]: i=2; j=2; k=0..2
    schoolbook_block(2, 2, 0, 1, 1, 3)
    # [1,1,4]: i=2; j=2; k=3..6
    schoolbook_block(2, 2, 3, 1, 1, 4)
    
    return {"U": U, "V": V, "W": W}
