"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = m  # 3
    
    U = [[0]*rank for _ in range(n*m)]  # shape (6, 3)
    V = [[0]*rank for _ in range(m*p)]  # shape (24, 3)
    W = [[0]*rank for _ in range(p*n)]  # shape (16, 3)
    
    # U[i*m+j, r] = delta(j, r)
    for i in range(n):
        for j in range(m):
            for r in range(rank):
                if j == r:
                    U[i*m+j][r] = 1
    
    # V[j*p+k, r] = delta(j, r)
    for j in range(m):
        for k in range(p):
            for r in range(rank):
                if j == r:
                    V[j*p+k][r] = 1
    
    # W[k*n+i, r] = 1 for all k, i, r
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W[k*n+i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
