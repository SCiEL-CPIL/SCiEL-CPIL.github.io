import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # We decompose into 3 blocks of 2x3 * 3x2
    # Rank of 2x3 * 3x2 is 6 (schoolbook)
    # Total rank = 3 * 6 = 18
    
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0] * rank for _ in range(p * n)]  # 12 x 18
    
    # Block size in p dimension is 2
    block_size = 2
    num_blocks = p // block_size  # 3
    
    col_offset = 0
    for t in range(num_blocks):
        # For each block t, we perform a 2x3 * 3x2 multiplication
        # Using schoolbook (6 terms)
        for i in range(n):
            for j in range(m):
                for k_local in range(block_size):
                    # Index in global V: j*p + (t*block_size + k_local)
                    # Index in global W: (t*block_size + k_local)*n + i
                    # Index in global U: i*m + j
                    
                    u_idx = i * m + j
                    v_idx = j * p + (t * block_size + k_local)
                    w_idx = (t * block_size + k_local) * n + i
                    
                    U[u_idx][col_offset] = 1
                    V[v_idx][col_offset] = 1
                    W[w_idx][col_offset] = 1
                    col_offset += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
