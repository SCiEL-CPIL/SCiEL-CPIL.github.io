"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    # Partition the 3x10 matrix into two 3x5 blocks.
    # Each 2x3x5 multiplication has schoolbook rank 15.
    # Total rank = 15 + 15 = 30.
    n, m, p = DIMENSIONS
    # p=10, split into two blocks of size 5
    block_p = 5
    num_blocks = p // block_p
    
    total_rank = n * m * block_p * num_blocks
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    column = 0
    for b in range(num_blocks):
        for i in range(n):
            for j in range(m):
                for k in range(block_p):
                    # U index: i*m + j
                    U[i * m + j][column] = 1
                    # V index: j*p + (b*block_p + k)
                    V[j * p + (b * block_p + k)][column] = 1
                    # W index: (b*block_p + k)*n + i
                    W[(b * block_p + k) * n + i][column] = 1
                    column += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
