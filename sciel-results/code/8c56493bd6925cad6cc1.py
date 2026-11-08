"""Tensor decomposition for 2x5 x 5x6 matrix multiplication, rank 54."""
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """Return rank-minimal tensor decomposition for 2x5 x 5x6 matmul.
    
    Uses Strassen-style 7-product scheme on 2x2 blocks for the first 4
    columns/rows, plus 12 scalar products for column 4.
    Total rank = 2*3*7 + 2*6 = 42 + 12 = 54.
    
    Convention: W[k*n + i, r] where i is the row index of C (0..n-1),
    k is the column index of C (0..p-1).
    U[i*m + j, r] where i is row of A, j is column of A.
    V[j*p + k, r] where j is row of B, k is column of B.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 5, 6
    rank = 54

    # Initialize factor matrices as lists of lists
    U = [[0] * rank for _ in range(n * m)]  # 10 x 54
    V = [[0] * rank for _ in range(m * p)]  # 30 x 54
    W = [[0] * rank for _ in range(p * n)]  # 12 x 54

    col = 0

    # Group the 5 columns of A (and 5 rows of B) into pairs + one leftover
    # Pairs: (0,1), (2,3); leftover: 4
    # For each pair (j0,j1) and each pair of output columns (k0,k1),
    # apply 7 Strassen products.
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]

    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # Product 1: (A[0,j0] + A[1,j1]) * (B[j0,k0] + B[j1,k1])
            # Contributes to C[0,k0], C[1,k1], and cross terms
            U[0 * m + j0][col] += 1
            U[1 * m + j1][col] += 1
            V[j0 * p + k0][col] += 1
            V[j1 * p + k1][col] += 1
            W[k0 * n + 0][col] += 1
            W[k1 * n + 1][col] += 1
            col += 1

            # Product 2: (A[1,j0] + A[1,j1]) * B[j0,k0]
            U[1 * m + j0][col] += 1
            U[1 * m + j1][col] += 1
            V[j0 * p + k0][col] += 1
            W[k0 * n + 1][col] += 1
            W[k1 * n + 1][col] -= 1
            col += 1

            # Product 3: A[0,j0] * (B[j0,k1] - B[j1,k1])
            U[0 * m + j0][col] += 1
            V[j0 * p + k1][col] += 1
            V[j1 * p + k1][col] -= 1
            W[k1 * n + 0][col] += 1
            W[k1 * n + 1][col] += 1
            col += 1

            # Product 4: A[1,j1] * (B[j1,k0] - B[j0,k0])
            U[1 * m + j1][col] += 1
            V[j1 * p + k0][col] += 1
            V[j0 * p + k0][col] -= 1
            W[k0 * n + 0][col] += 1
            W[k0 * n + 1][col] += 1
            col += 1

            # Product 5: (A[0,j0] + A[0,j1]) * B[j1,k1]
            U[0 * m + j0][col] += 1
            U[0 * m + j1][col] += 1
            V[j1 * p + k1][col] += 1
            W[k0 * n + 0][col] -= 1
            W[k1 * n + 0][col] += 1
            col += 1

            # Product 6: (A[1,j0] - A[0,j0]) * (B[j0,k0] + B[j0,k1])
            U[1 * m + j0][col] += 1
            U[0 * m + j0][col] -= 1
            V[j0 * p + k0][col] += 1
            V[j0 * p + k1][col] += 1
            W[k1 * n + 1][col] += 1
            col += 1

            # Product 7: (A[0,j1] - A[1,j1]) * (B[j1,k0] + B[j1,k1])
            U[0 * m + j1][col] += 1
            U[1 * m + j1][col] -= 1
            V[j1 * p + k0][col] += 1
            V[j1 * p + k1][col] += 1
            W[k0 * n + 0][col] += 1
            col += 1

    # Handle the leftover column j=4 with scalar products
    for k in range(p):
        # C[0,k] += A[0,4] * B[4,k]
        U[0 * m + 4][col] += 1
        V[4 * p + k][col] += 1
        W[k * n + 0][col] += 1
        col += 1

        # C[1,k] += A[1,4] * B[4,k]
        U[1 * m + 4][col] += 1
        V[4 * p + k][col] += 1
        W[k * n + 1][col] += 1
        col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    if time.time() > deadline:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
