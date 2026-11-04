<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using Kronecker product structure M_{n,1,1} \otimes M_{1,m,p}
    # Rank is r(M_{n,1,1}) * r(M_{1,m,p})
    # r(M_{n,1,1}) = n
    # r(M_{1,m,p}) = m*p (schoolbook)
    # Total rank = n * m * p
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i (Trace convention)
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Decompose M_{n,m,p} as M_{n,1,1} \otimes M_{1,m,p}
    # Rank(M_{n,1,1}) = n
    # Rank(M_{1,m,p}) = m*p
    # Total Rank = n * m * p
    
    rank = n * m * p
    rank1 = n
    rank2 = m * p
    
    # M_{n,1,1} factors
    # U1: (n*1, n), V1: (1*1, n), W1: (1*n, n)
    U1 = [[0] * rank1 for _ in range(n)]
    V1 = [[0] * rank1 for _ in range(1)]
    W1 = [[0] * rank1 for _ in range(1)]
    
    for i in range(n):
        U1[i][i] = 1
        V1[0][i] = 1
        W1[0][i] = 1
        
    # M_{1,m,p} factors
    # U2: (1*m, m*p), V2: (m*p, m*p), W2: (p*1, m*p)
    U2 = [[0] * rank2 for _ in range(m)]
    V2 = [[0] * rank2 for _ in range(m * p)]
    W2 = [[0] * rank2 for _ in range(p)]
    
    for j in range(m):
        for k in range(p):
            r2 = j * p + k
            U2[j][r2] = 1
            V2[j * p + k][r2] = 1
            W2[k][r2] = 1
            
    # Kronecker Products
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U = U1 \otimes U2
    for i1 in range(n):
        for r1 in range(rank1):
            if U1[i1][r1] == 0: continue
            for i2 in range(m):
                for r2 in range(rank2):
                    if U2[i2][r2] == 0: continue
                    r = r1 * rank2 + r2
                    U[i1 * m + i2][r] = U1[i1][r1] * U2[i2][r2]
                    
    # V = V1 \otimes V2
    for i1 in range(1):
        for r1 in range(rank1):
            if V1[i1][r1] == 0: continue
            for i2 in range(m * p):
                for r2 in range(rank2):
                    if V2[i2][r2] == 0: continue
                    r = r1 * rank2 + r2
                    V[i1 * (m * p) + i2][r] = V1[i1][r1] * V2[i2][r2]
                    
    # W = W1 \otimes W2
    for i1 in range(1):
        for r1 in range(rank1):
            if W1[i1][r1] == 0: continue
            for i2 in range(p):
                for r2 in range(rank2):
                    if W2[i2][r2] == 0: continue
                    r = r1 * rank2 + r2
                    # W index in full tensor: k*n + i
                    # i1 is index for W1 (row 0). r1 is the 'i' from M_{n,1,1}
                    # i2 is index for W2 (row k). r2 is the 'k' from M_{1,m,p}
                    # In M_{n,1,1}, W1[0, r1] corresponds to i=r1.
                    # In M_{1,m,p}, W2[i2, r2] corresponds to k=i2.
                    # So full W index is i2 * n + r1.
                    W[i2 * n + r1][r] = W1[i1][r1] * W2[i2][r2]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
