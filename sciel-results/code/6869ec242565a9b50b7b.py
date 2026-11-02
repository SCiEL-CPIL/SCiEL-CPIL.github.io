"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Kronecker-structured decomposition:
    2x3 * 3x9 = 2x9
    
    Decompose the 3x9 multiplication into three parallel 3x3 blocks:
    A (2x3) * B (3x9) where B = [B0 | B1 | B2], each Bq is 3x3.
    
    For each block q, we need to compute A * Bq (2x3 * 3x3 -> 2x3).
    The tensor for 2x3 * 3x3 has rank 6 (schoolbook).
    
    Total rank = 3 blocks * 6 per block = 18.
    
    Index conventions:
    U[i*m + j, c] where i in [0,n), j in [0,m), c in [0,rank)
    V[j*p + k, c] where j in [0,m), k in [0,p), c in [0,rank)
    W[k*n + i, c] where k in [0,p), i in [0,n), c in [0,rank)
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    block_size = 3  # p is multiple of 3, so 3 blocks of size 3
    
    # Rank-6 decomposition for 2x3 * 3x3 (schoolbook)
    # For each (i, j, k) in [0,2)x[0,3)x[0,3):
    #   U[i*3+j] = 1, V[j*3+k] = 1, W[k*2+i] = 1
    
    num_blocks = p // block_size  # 3 blocks
    rank_per_block = n * m * block_size  # 2*3*3 = 6
    total_rank = num_blocks * rank_per_block  # 18
    
    U = [[0] * total_rank for _ in range(n * m)]      # 6 x 18
    V = [[0] * total_rank for _ in range(m * p)]      # 27 x 18
    W = [[0] * total_rank for _ in range(p * n)]      # 18 x 18
    
    for q in range(num_blocks):
        base_col = q * rank_per_block
        for i in range(n):
            for j in range(m):
                for k in range(block_size):
                    col = base_col + (i * m + j) * block_size + k
                    # U: row i*m+j, col
                    U[i * m + j][col] = 1
                    # V: row j*p + (q*block_size + k), col
                    V[j * p + q * block_size + k][col] = 1
                    # W: row (q*block_size + k)*n + i, col
                    W[(q * block_size + k) * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
