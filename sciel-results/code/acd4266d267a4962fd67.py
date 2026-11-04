"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 9 columns of B into 4 pairs and 1 single column.
    # Each pair corresponds to a 2x3 by 3x2 multiplication (schoolbook rank 12).
    # The single column corresponds to a 2x3 by 3x1 multiplication (rank 2).
    # Total rank = 4 * 12 + 2 = 50.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 4 pairs of columns: (0,1), (2,3), (4,5), (6,7)
    for pair_idx in range(4):
        k0 = pair_idx * 2
        k1 = pair_idx * 2 + 1
        for i in range(n):
            for j in range(m):
                for kk in range(2): # 0 for k0, 1 for k1
                    k = k0 + kk
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
    # 1 single column: 8
    k = 8
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
