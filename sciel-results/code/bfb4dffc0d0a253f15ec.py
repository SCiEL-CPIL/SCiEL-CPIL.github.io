"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 4*6 = 24
    U = [[0] * rank for _ in range(n * m)]  # (8, 24)
    V = [[0] * rank for _ in range(m * p)]  # (24, 24)
    W = [[0] * rank for _ in range(p * n)]  # (12, 24)
    
    col = 0
    for j in range(m):
        for k in range(p):
            # U: pick A(i,j) for i=0,1
            for i in range(n):
                U[i * m + j][col] = 1
            # V: pick B(j,k)
            V[j * p + k][col] = 1
            # W: distribute to C(i,k) for i=0,1
            for i in range(n):
                W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
