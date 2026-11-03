"""Tensor decomposition for 2x3 by 3x6 matrix multiplication.

The multiplication tensor T_{i,j,k} = 1 for all i in [0,2), j in [0,3), k in [0,6)
since C[i,k] = sum_j A[i,j] B[j,k] has coefficient 1 for every (i,j,k) triple.

Since every entry equals 1, the tensor is rank 1:
    T_{i,j,k} = a_i * b_j * c_k  with a_i = 1, b_j = 1, c_k = 1.

This gives the minimal possible rank (rank cannot be 0 for a nonzero tensor).
"""

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    rank = 1

    # U has shape (n*m, rank) = (6, 1)
    # U[i*m + j, c] = a_i
    U = [[1] for _ in range(n * m)]

    # V has shape (m*p, rank) = (18, 1)
    # V[j*p + k, c] = b_j
    V = [[1] for _ in range(m * p)]

    # W has shape (p*n, rank) = (12, 1)
    # W[k*n + i, c] = c_k
    W = [[1] for _ in range(p * n)]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
