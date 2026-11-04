import numpy as np

def _get_block_factors(n, m, p):
    # Returns U (n*m x rank), V (m*p x rank), W (p*n x rank)
    
    if n == 2 and m == 2 and p == 2:
        # Strassen
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M0
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1
        
        # M2
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = -1
        
        # M3
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[1][3] = 1; W[0][3] = -1
        
        # M4
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[2][4] = 1
        
        # M5
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[0][5] = 1
        
        # M6
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[3][6] = 1
        
        return U, V, W

    elif n == 2 and m == 2 and p == 1:
        # Rank 2
        rank = 2
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]
        
        # c=0: a=0, b=0
        U[0][0] = 1 # i=0, a=0
        U[2][0] = 1 # i=1, a=0
        V[0][0] = 1 # b=0
        W[0][0] = 1 # k=0, i=0
        W[1][0] = 1 # k=0, i=1
        
        # c=1: a=1, b=1
        U[1][1] = 1 # i=0, a=1
        U[3][1] = 1 # i=1, a=1
        V[1][1] = 1 # b=1
        W[0][1] = 1 # k=0, i=0
        W[1][1] = 1 # k=0, i=1
        
        return U, V, W

    elif n == 2 and m == 1 and p == 2:
        # Rank 1
        rank = 1
        U = [[0]*rank for _ in range(2)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]
        
        U[0][0] = 1
        U[1][0] = 1
        V[0][0] = 1
        V[1][0] = 1
        W[0][0] = 1 # k=0, i=0
        W[1][0] = 1 # k=1, i=0
        W[2][0] = 1 # k=0, i=1
        W[3][0] = 1 # k=1, i=1
        
        return U, V, W

    elif n == 2 and m == 1 and p == 1:
        # Rank 1
        rank = 1
        U = [[0]*rank for _ in range(2)]
        V = [[0]*rank for _ in range(1)]
        W = [[0]*rank for _ in range(2)]
        
        U[0][0] = 1
        U[1][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        W[1][0] = 1
        
        return U, V, W

    elif n == 1 and m == 2 and p == 2:
        # Rank 2
        rank = 2
        U = [[0]*rank for _ in range(2)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(2)]
        
        # c=0: a=0, b=0
        U[0][0] = 1 # a=0
        V[0][0] = 1 # b=0, k=0
        V[2][0] = 1 # b=0, k=1
        W[0][0] = 1 # k=0, i=0
        W[1][0] = 1 # k=1, i=0
        
        # c=1: a=1, b=1
        U[1][1] = 1 # a=1
        V[1][1] = 1 # b=1, k=0
        V[3][1] = 1 # b=1, k=1
        W[0][1] = 1 # k=0, i=0
        W[1][1] = 1 # k=1, i=0
        
        return U, V, W

    elif n == 1 and m == 2 and p == 1:
        # Rank 2
        rank = 2
        U = [[0]*rank for _ in range(2)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(1)]
        
        # c=0: a=0, b=0
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        
        # c=1: a=1, b=1
        U[1][1] = 1
        V[1][1] = 1
        W[0][1] = 1
        
        return U, V, W

    elif n == 1 and m == 1 and p == 2:
        # Rank 1
        rank = 1
        U = [[0]*rank for _ in range(1)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(2)]
        
        U[0][0] = 1
        V[0][0] = 1
        V[1][0] = 1
        W[0][0] = 1
        W[1][0] = 1
        
        return U, V, W

    elif n == 1 and m == 1 and p == 1:
        # Rank 1
        rank = 1
        U = [[0]*rank for _ in range(1)]
        V = [[0]*rank for _ in range(1)]
        W = [[0]*rank for _ in range(1)]
        
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        
        return U, V, W

    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i*m+a][c] = 1
                    V[a*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return U, V, W

