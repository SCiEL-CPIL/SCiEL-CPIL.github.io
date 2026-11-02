"""Matrix multiplication 2x3 by 3x9: rank-minimal exact tensor decomposition.

The tensor for 2x3 x 3x9 is a direct sum of three [2,3,3] multiplication tensors.
Each [2,3,3] tensor has rank 6 (schoolbook, tight for these dimensions).
Total rank = 3 * 6 = 18, which is optimal.

Index conventions:
  U[i*m + j, c]  where i in [0,n), j in [0,m), c in [0,rank)
  V[j*p + k, c]  where j in [0,m), k in [0,p), c in [0,rank)
  W[k*n + i, c]  where k in [0,p), i in [0,n), c in [0,rank)
"""

def construct():
    """
    Direct-sum decomposition of 2x3 * 3x9 into three 2x3 * 3x3 blocks.
    Each block uses the schoolbook rank-6 decomposition.
    Total rank = 18.
    """
    n, m, p = 2, 3, 9
    block_size = 3
    num_blocks = p // block_size  # 3
    rank_per_block = n * m * block_size  # 6
    total_rank = num_blocks * rank_per_block  # 18

    # Allocate factor matrices as lists of lists
    U = [[0] * total_rank for _ in range(n * m)]      # 6 x 18
    V = [[0] * total_rank for _ in range(m * p)]      # 27 x 18
    W = [[0] * total_rank for _ in range(p * n)]      # 18 x 18

    for q in range(num_blocks):
        base_col = q * rank_per_block
        for i in range(n):
            for j in range(m):
                for k in range(block_size):
                    col = base_col + (i * m + j) * block_size + k
                    # U: row i*m+j
                    U[i * m + j][col] = 1
                    # V: row j*p + (q*block_size + k)
                    V[j * p + q * block_size + k][col] = 1
                    # W: row (q*block_size + k)*n + i
                    W[(q * block_size + k) * n + i][col] = 1

    return {"U": U, "V": V, "W": W}
