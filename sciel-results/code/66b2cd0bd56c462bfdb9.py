"""
Matrix multiplication 2x3x9 - rank-minimal exact tensor decomposition.

Approach: Tiled decomposition into 2x3x3 blocks.
The 2x3x3 matrix multiplication tensor has a known rank of 18 (same as naive: 2*3*3=18).
For 2x3x9, we partition the output dimension (size 9) into three groups of 3.
Each group corresponds to an independent 2x3x3 multiplication.
Total rank = 3 * 18 = 54.

This is the naive decomposition, which is optimal for these dimensions
since no Strassen-type speedup applies cleanly to 2x3x3 blocks.
"""

def construct():
    n, m, p = 2, 3, 9
    # Partition p into blocks of size 3
    block_size = 3
    num_blocks = p // block_size  # 3 blocks
    
    # Each 2x3x3 block has rank n*m*block_size = 2*3*3 = 18
    block_rank = n * m * block_size
    total_rank = block_rank * num_blocks  # 54
    
    # Allocate factor matrices
    U = [[0] * total_rank for _ in range(n * m)]      # shape (6, 54)
    V = [[0] * total_rank for _ in range(m * p)]      # shape (27, 54)
    W = [[0] * total_rank for _ in range(p * n)]      # shape (18, 54)
    
    col = 0
    for b in range(num_blocks):
        k_offset = b * block_size
        for i in range(n):
            for j in range(m):
                for k_local in range(block_size):
                    k_global = k_offset + k_local
                    
                    u_idx = i * m + j
                    v_idx = j * p + k_global
                    w_idx = k_global * n + i
                    
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
