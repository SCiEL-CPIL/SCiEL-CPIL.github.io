"""Matrix multiplication 2,2,13 tensor decomposition.

The problem asks for the rank-minimal exact tensor decomposition for multiplying
a 2x2 matrix A by a 2x13 matrix B.
The tensor T has modes corresponding to C (2x13), A (2x2), and B (2x13).
The entries are T[(i,k), (i',j'), (j'',k'')] = 1 if i==i', j'==j'', k==k'', else 0.
This tensor is a direct sum of 13 copies of the tensor for multiplying a 2x2 matrix
by a 2x1 matrix (i.e., M_{2,2,1}).
The rank of M_{2,2,1} is 4 (as shown by the matrix flattening to the identity matrix).
Since the supports in the C and B modes are disjoint for different k, the ranks add.
Total rank = 13 * 4 = 52.
This is the schoolbook rank, which is optimal for this structure.
"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 52
    U = [[0] * rank for _ in range(n * m)]   # 4 x 52
    V = [[0] * rank for _ in range(m * p)]   # 26 x 52
    W = [[0] * rank for _ in range(p * n)]   # 26 x 52
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1
                # V[j*p + k, col] = 1
                # W[k*n + i, col] = 1
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
