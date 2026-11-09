"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Kronecker product decomposition for 3x4 x 4x8 matrix multiplication.
    
    Factor: n=3*1, m=2*2, p=4*2
    Sub-problem 1: (3x2) x (2x4) - schoolbook, rank 24
    Sub-problem 2: (1x2) x (2x2) - schoolbook, rank 4
    Total rank: 24 * 4 = 96
    
    This is equivalent to direct schoolbook but constructed via Kronecker product.
    """
    n, m, p = DIMENSIONS
    
    # Sub-problem 1: (n1 x m1) x (m1 x p1) = (3 x 2) x (2 x 4)
    n1, m1, p1 = 3, 2, 4
    # Sub-problem 2: (n2 x m2) x (m2 x p2) = (1 x 2) x (2 x 2)
    n2, m2, p2 = 1, 2, 2
    
    # Schoolbook decomposition for sub-problem 1
    rank1 = n1 * m1 * p1
    U1 = [[0] * rank1 for _ in range(n1 * m1)]
    V1 = [[0] * rank1 for _ in range(m1 * p1)]
    W1 = [[0] * rank1 for _ in range(p1 * n1)]
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j][col] = 1
                V1[j * p1 + k][col] = 1
                W1[k * n1 + i][col] = 1
                col += 1
    
    # Schoolbook decomposition for sub-problem 2
    rank2 = n2 * m2 * p2
    U2 = [[0] * rank2 for _ in range(n2 * m2)]
    V2 = [[0] * rank2 for _ in range(m2 * p2)]
    W2 = [[0] * rank2 for _ in range(p2 * n2)]
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j][col] = 1
                V2[j * p2 + k][col] = 1
                W2[k * n2 + i][col] = 1
                col += 1
    
    # Kronecker product: rank = rank1 * rank2
    rank = rank1 * rank2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r1 in range(rank1):
        for r2 in range(rank2):
            r = r1 * rank2 + r2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            # U index: (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
                            u_idx = (i1 * n2 + i2) * (m1 * m2) + (j1 * m2 + j2)
                            U[u_idx][r] = U1[i1 * m1 + j1][r1] * U2[i2 * m2 + j2][r2]
            
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            # V index: (j1*m2 + j2) * (p1*p2) + (k1*p2 + k2)
                            v_idx = (j1 * m2 + j2) * (p1 * p2) + (k1 * p2 + k2)
                            V[v_idx][r] = V1[j1 * p1 + k1][r1] * V2[j2 * p2 + k2][r2]
            
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            # W index: (k1*p2 + k2) * (n1*n2) + (i1*n2 + i2)
                            w_idx = (k1 * p2 + k2) * (n1 * n2) + (i1 * n2 + i2)
                            W[w_idx][r] = W1[k1 * n1 + i1][r1] * W2[k2 * n2 + i2][r2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
