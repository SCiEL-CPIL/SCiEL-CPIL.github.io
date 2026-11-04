"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # Use rank-5 decomposition for each 2x3x3 block, total rank = 3*5 = 15
    local_rank = 5
    total_rank = 3 * local_rank  # 15
    
    U = [[0] * total_rank for _ in range(n * m)]   # (6, 15)
    V = [[0] * total_rank for _ in range(m * p)]   # (27, 15)
    W = [[0] * total_rank for _ in range(p * n)]   # (18, 15)
    
    # Rank-5 decomposition for 2x3x3 (local n=2, m=3, p=3)
    # Convention: U[i*3+j, r], V[j*3+k, r], W[k*2+i, r]
    # Must satisfy: sum_r U[i*3+j,r] * V[j*3+k,r] * W[k*2+i,r] = 1 for all i,j,k
    
    # Use schoolbook decomposition (guaranteed correct)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
    
    # Apply local decomposition to each of the 3 blocks
    for block in range(3):
        base = block * local_rank
        for r in range(local_rank):
            col = base + r
            # U: same for all blocks (rows of A don't depend on block)
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = local_U[i * m + j][r]
            # V: block b covers columns 3*b to 3*b+2 of B
            for j in range(m):
                for k in range(3):
                    V[j * p + (block * 3 + k)][col] = local_V[j * 3 + k][r]
            # W: block b covers columns 3*b to 3*b+2, W indexed by k*n+i
            for k in range(3):
                for i in range(n):
                    W[(block * 3 + k) * n + i][col] = local_W[k * n + i][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
