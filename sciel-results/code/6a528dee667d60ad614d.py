"""
Rank-48 decomposition for 3x4 * 4x4 matrix multiplication.

Approach: Row-Decoupled Decomposition.
The matrix multiplication $C = AB$ where $A$ is $3 \times 4$ and $B$ is $4 \times 4$ results in $C$ being $3 \times 4$.
The $i$-th row of $C$ is given by $C_{i,:} = A_{i,:} B$.
This operation depends only on the $i$-th row of $A$ and the entire matrix $B$.
Since the rows of $A$ are disjoint in the input tensor, the total tensor is the direct sum of three tensors, each corresponding to a $1 \times 4 \times 4$ multiplication.
The rank of a $1 \times m \times p$ multiplication tensor is $m \times p$.
Here $m=4, p=4$, so each row requires rank 16.
Total rank = $3 \times 16 = 48$.
This matches the schoolbook rank but is structured differently (grouped by output rows).
"""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = n * m * p  # 48
    U = [[0] * rank for _ in range(n * m)]  # 12 x 48
    V = [[0] * rank for _ in range(m * p)]  # 16 x 48
    W = [[0] * rank for _ in range(p * n)]  # 12 x 48

    col = 0
    # Iterate over each row of the output (and corresponding row of A)
    for i in range(n):
        # For each row i, we compute C[i, :] = A[i, :] * B
        # This is a 1 x 4 x 4 multiplication.
        # A indices: i*m + j
        # B indices: j*p + k
        # C indices: k*n + i
        for j in range(m):
            for k in range(p):
                # U index for A[i, j]
                u_idx = i * m + j
                # V index for B[j, k]
                v_idx = j * p + k
                # W index for C[k, i]
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
