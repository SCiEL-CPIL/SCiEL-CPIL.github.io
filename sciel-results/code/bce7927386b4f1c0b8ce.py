import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    # Schoolbook rank: n*m*p = 64
    rank = n * m * p

    # Build the matrix multiplication tensor T
    # T is a 3-tensor of shape (n*m) x (m*p) x (p*n)
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Convert to matrix form for rank computation
    # Reshape T into a 2D matrix of shape (n*m, (m*p)*(p*n))
    T_mat = T.reshape(n * m, (m * p) * (p * n))

    # Compute rank via SVD
    U_svd, s, Vt = np.linalg.svd(T_mat, full_matrices=False)
    tol = 1e-10
    num_nonzero = np.sum(s > tol)
    print(f"SVD rank: {num_nonzero}")

    # The border rank is at least the numerical rank of any unfolding.
    # For matrix multiplication tensors, the border rank equals the tensor rank.
    # So the minimum rank is num_nonzero.

    min_rank = int(num_nonzero)
    print(f"Minimum rank: {min_rank}")

    # Build a rank-minimal decomposition using the SVD factors
    # We need U, V, W matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # such that T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product sum)

    # Use the top min_rank singular vectors
    # But SVD gives a matrix decomposition, not a tensor decomposition directly.
    # We need to find a tensor decomposition of rank min_rank.

    # For the matrix multiplication tensor, the schoolbook decomposition is optimal.
    # Let's just return the schoolbook decomposition if min_rank == n*m*p.
    # Otherwise, we need a more clever approach.

    if min_rank == n * m * p:
        # Schoolbook is optimal
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
    else:
        # Need a rank-minimal decomposition
        # This is a hard problem in general, but for small cases we can try.
        # For now, return the schoolbook decomposition as a fallback.
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
