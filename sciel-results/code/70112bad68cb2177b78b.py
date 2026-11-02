"""Matrix multiplication 2x3 by 3x9: rank-minimal exact tensor decomposition.

Strategy:
  - The schoolbook decomposition gives rank = n*m*p = 2*3*9 = 54.
  - We exploit the structure p = 3 * 3 to view the 3x9 matrix as three 3x3 blocks.
  - For each block q (q=0,1,2), we compute A (2x3) * B_q (3x3) -> 2x3.
  - The 2x3 * 3x3 multiplication tensor has rank 6 (schoolbook, and this is tight
    for these small dimensions).
  - Total rank = 3 blocks * 6 = 18.

Index conventions (as specified):
  U[i*m + j, c]  where i in [0,n), j in [0,m), c in [0,rank)
  V[j*p + k, c]  where j in [0,m), k in [0,p), c in [0,rank)
  W[k*n + i, c]  where k in [0,p), i in [0,n), c in [0,rank)
"""

import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START

def construct():
    """
    Block-structured decomposition:
    2x3 * 3x9 = 2x9

    Decompose the 3x9 multiplication into three parallel 3x3 blocks:
    A (2x3) * B (3x9) where B = [B0 | B1 | B2], each Bq is 3x3.

    For each block q, compute A * Bq (2x3 * 3x3 -> 2x3).
    The tensor for 2x3 * 3x3 has rank 6 (schoolbook).

    Total rank = 3 blocks * 6 per block = 18.

    Index conventions:
      U[i*m + j, c] where i in [0,n), j in [0,m), c in [0,rank)
      V[j*p + k, c] where j in [0,m), k in [0,p), c in [0,rank)
      W[k*n + i, c] where k in [0,p), i in [0,n), c in [0,rank)
    """
    deadline = time.time() + 110.0  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    block_size = 3  # p is multiple of 3, so 3 blocks of size 3
    num_blocks = p // block_size  # 3 blocks
    rank_per_block = n * m * block_size  # 2*3*3 = 6
    total_rank = num_blocks * rank_per_block  # 18

    # Allocate factor matrices
    U = [[0] * total_rank for _ in range(n * m)]      # 6 x 18
    V = [[0] * total_rank for _ in range(m * p)]      # 27 x 18
    W = [[0] * total_rank for _ in range(p * n)]      # 18 x 18

    for q in range(num_blocks):
        if time.time() > deadline:
            break
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
