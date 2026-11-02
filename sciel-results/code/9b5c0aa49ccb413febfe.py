import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Approach: Block decomposition of the p-dimension into blocks of size 3.
    # Each block corresponds to a 2x3x3 matrix multiplication.
    # The trivial rank of 2x3x3 is 18.
    # Total rank is 3 * 18 = 54.
    
    block_size = 3
    num_blocks = p // block_size
    rank_per_block = n * m * block_size
    total_rank = num_blocks * rank_per_block
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for t in range(num_blocks):
        for i in range(n):
            for j in range(m):
                for s in range(block_size):
                    k = t * block_size + s
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
                    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
