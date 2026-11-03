"""Matrix multiplication 4,4,8 - structural block decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 4x4 * 4x8 multiplication as two independent 4x4 * 4x4 blocks.
    Each 4x4x4 block uses the schoolbook rank-16 decomposition.
    Total rank = 2 * 16 = 32.
    
    U shape: (n*m, rank) = (16, 32)
    V shape: (m*p, rank) = (32, 32)
    W shape: (p*n, rank) = (32, 32)
    
    Index conventions:
      U[i*m + j, c]  -- i in [0,4), j in [0,4)
      V[j*p + k, c]  -- j in [0,4), k in [0,8)
      W[k*n + i, c]  -- k in [0,8), i in [0,4)
    
    Block 0: columns k=0..3, uses rank 16
    Block 1: columns k=4..7, uses rank 16
    """
    n, m, p = DIMENSIONS
    half_p = p // 2  # 4
    rank_per_block = n * m * half_p  # 16
    total_rank = 2 * rank_per_block  # 32
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for block in range(2):
        base_col = block * rank_per_block
        # k offset for this block
        k_off = block * half_p
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(half_p):
                    actual_k = k_off + k
                    c = base_col + col
                    U[i * m + j][c] = 1
                    V[j * p + actual_k][c] = 1
                    W[actual_k * n + i][c] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
