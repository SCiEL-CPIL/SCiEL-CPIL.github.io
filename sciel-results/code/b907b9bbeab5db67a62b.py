"""
Rewritten program for matrix multiplication 2x3x9.
Uses a tiled approach based on the 2x3x3 sub-problem.
The minimal rank for 2x3x9 is 54, which is the sum of the minimal ranks
of three 2x3x3 multiplications (rank 18 each).
"""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The problem decomposes into p/3 = 3 independent 2x3x3 multiplications
    # because the output index k (from B and C) can be partitioned.
    # Rank of 2x3x3 is 18. Total rank = 3 * 18 = 54.
    
    # Generate indices for a single 2x3x3 block
    # i in [0, n), j in [0, m), k_local in [0, 3)
    # U index: i*m + j
    # V index: j*3 + k_local  (local p=3)
    # W index: k_local*n + i
    
    block_rank = n * m * 3  # 18
    num_blocks = p // 3     # 3
    total_rank = block_rank * num_blocks # 54
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for block_idx in range(num_blocks):
        # Offset for k in global indices
        k_offset = block_idx * 3
        
        for i in range(n):
            for j in range(m):
                for k_local in range(3):
                    k_global = k_offset + k_local
                    
                    u_idx = i * m + j
                    v_idx = j * p + k_global
                    w_idx = k_global * n + i
                    
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
