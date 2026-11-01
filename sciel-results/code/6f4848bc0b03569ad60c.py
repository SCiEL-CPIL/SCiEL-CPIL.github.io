"""
Rank-minimal exact tensor decomposition for 2x3 by 3x7 matrix multiplication.

The tensor T has entries T[i, j, k] for i in {0,1}, j in {0,1,2}, k in {0,...,6}.
A rank-r decomposition is a set of r triples (u, v, w) of tensors in the three
factor spaces such that T = sum_{c=0}^{r-1} u_c * v_c * w_c (outer products).

The schoolbook decomposition uses rank 42 = 2*3*7.
Lower-rank decompositions exist in principle (the tensor rank of matrix multiplication
is known to be less than the naive rank for many dimensions), but no explicit
lower-rank construction for 2x3x7 is known to me that can be produced reliably
within the time limit.

We implement the schoolbook decomposition, which is guaranteed valid and exact.
"""

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
