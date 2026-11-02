"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # Decompose m=4 into two blocks of size 2
    # Total rank = 2 blocks * 2 (j_local) * 8 (k) = 32
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 8 x 32
    V = [[0] * rank for _ in range(m * p)]  # 32 x 32
    W = [[0] * rank for _ in range(p * n)]  # 16 x 32
    
    col = 0
    for b in range(2):  # two blocks
        for j_local in range(2):  # two rows in each 2x2 block
            j = 2 * b + j_local
            for k in range(p):  # 8 columns
                # U: rows i*4 + j for i=0,1
                for i in range(n):
                    U[i * m + j][col] = 1
                # V: row j*p + k
                V[j * p + k][col] = 1
                # W: rows k*n + i for i=0,1
                for i in range(n):
                    W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
