"""
Alternative approach: Decomposition into independent outer product tensors.
The matrix multiplication C = AB where A is 2x3 and B is 3x9 can be viewed as
C[:, k] = A[:, 0] * B[0, k] + A[:, 1] * B[1, k] + A[:, 2] * B[2, k].
This is a sum of 3 terms, each of which is an outer product of a vector from A
and a vector from B.
Specifically, for each column j of A (which is a 2x1 vector) and the corresponding
row j of B (which is a 1x9 vector), we form an outer product.
The tensor for a 2x1x9 outer product has rank 2*9 = 18.
Since the B-indices for different j are disjoint, the total rank is 3 * 18 = 54.
"""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank for each j-block (outer product 2x1x9) is n*p
    rank_per_block = n * p
    total_rank = m * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        # For each j, we decompose the outer product A[:, j] * B[j, :]^T
        # This corresponds to terms u_{i, j} * v_{j, k} * w_{k, i}
        # where i in [0, n), k in [0, p)
        for i in range(n):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
