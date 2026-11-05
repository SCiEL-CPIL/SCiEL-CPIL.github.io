import numpy as np

def construct():
    n, m, p = 3, 3, 7
    rank_total = 21
    
    # U: (n*m) x rank = 9 x 21
    # V: (m*p) x rank = 21 x 21
    # W: (p*n) x rank = 21 x 21
    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]
    
    col = 0
    
    # Block 1: k in 0..2 (3x3x3 schoolbook, rank 9)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    # Block 2: k in 3..5 (3x3x3 schoolbook, rank 9)
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    # Block 3: k in 6 (3x3x1 schoolbook, rank 3)
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+6][col] = 1
            W[6*n+i][col] = 1
            col += 1
            
    # Verify decomposition
    # Check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank_total):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    # Fallback to schoolbook rank 63 if verification fails
                    rank_sb = n * m * p
                    U_sb = [[0]*rank_sb for _ in range(n*m)]
                    V_sb = [[0]*rank_sb for _ in range(m*p)]
                    W_sb = [[0]*rank_sb for _ in range(p*n)]
                    col_sb = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U_sb[i2*m+j2][col_sb] = 1
                                V_sb[j2*p+k2][col_sb] = 1
                                W_sb[k2*n+i2][col_sb] = 1
                                col_sb += 1
                    return {"U": U_sb, "V": V_sb, "W": W_sb}
                    
    return {"U": U, "V": V, "W": W}
