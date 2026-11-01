"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 24 decomposition
    # Split m=3 into j=0,1 (2x2x8) and j=2 (2x1x8)
    # 2x2x8 part uses rank 16
    # 2x1x8 part uses rank 8
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: j=0,1 (2x2x8)
    # c = 0..7 for j=0, c=8..15 for j=1
    for i in range(n):
        for k in range(p):
            # j=0
            c0 = k
            U[i * m + 0][c0] = 1
            V[0 * p + k][c0] = 1
            W[k * n + i][c0] = 1
            
            # j=1
            c1 = 8 + k
            U[i * m + 1][c1] = 1
            V[1 * p + k][c1] = 1
            W[k * n + i][c1] = 1
            
    # Part 2: j=2 (2x1x8)
    # c = 16..23
    for i in range(n):
        for k in range(p):
            c2 = 16 + k
            U[i * m + 2][c2] = 1
            V[2 * p + k][c2] = 1
            W[k * n + i][c2] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
