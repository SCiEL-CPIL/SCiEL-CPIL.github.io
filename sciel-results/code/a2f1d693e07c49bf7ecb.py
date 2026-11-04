def construct():
    n, m, p = 4, 5, 6
    nm, mp, pn = n*m, m*p, p*n
    
    def strassen2():
        # Returns U(4x7), V(4x7), W(4x7) for 2x2x2
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[2][1]=1; W[2][1]=1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
        return U, V, W, 7
    
    def block_decomp(ni, mi, pi):
        """Return (U, V, W, rank) for ni x mi x pi block"""
        if ni == 2 and mi == 2 and pi == 2:
            return strassen2()
        elif mi == 1:
            # Rank = pi
            rank = pi
            U = [[0]*rank for _ in range(ni)]
            V = [[0]*rank for _ in range(pi)]
            W = [[0]*rank for _ in range(pi*ni)]
            for k in range(pi):
                for i in range(ni):
                    U[i][k] = 1
                    V[k][k] = 1
                    W[k*ni+i][k] = 1
            return U, V, W, rank
        elif pi == 1:
            # Rank = mi
            rank = mi
            U = [[0]*rank for _ in range(ni*mi)]
            V = [[0]*rank for _ in range(mi)]
            W = [[0]*rank for _ in range(ni)]
            for j in range(mi):
                for i in range(ni):
                    U[i*mi+j][j] = 1
                    V[j][j] = 1
                    W[i][j] = 1
            return U, V, W, rank
        elif ni == 1:
            # Rank = mi
            rank = mi
            U = [[0]*rank for _ in range(mi)]
            V = [[0]*rank for _ in range(mi*pi)]
            W = [[0]*rank for _ in range(pi)]
            for j in range(mi):
                for k in range(pi):
                    U[j][j] = 1
                    V[j*pi+k][j] = 1
                    W[k][j] = 1
            return U, V, W, rank
        else:
            # Schoolbook
            rank = ni * mi * pi
            U = [[0]*rank for _ in range(ni*mi)]
            V = [[0]*rank for _ in range(mi*pi)]
            W = [[0]*rank for _ in range(pi*ni)]
            col = 0
            for i in range(ni):
                for j in range(mi):
                    for k in range(pi):
                        U[i*mi+j][col] = 1
                        V[j*pi+k][col] = 1
                        W[k*ni+i][col] = 1
                        col += 1
            return U, V, W, rank
