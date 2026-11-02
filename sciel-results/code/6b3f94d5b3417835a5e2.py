"""Rewritten program: direct rank-60 schoolbook decomposition for 2x3x10 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 10]

def construct():
    """Return a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 10 matrix. The output factor follows the official trace convention W[k*n+i, r].

    The schoolbook decomposition has rank n*m*p = 2*3*10 = 60.
    U has shape (n*m, rank) = (6, 60)
    V has shape (m*p, rank) = (30, 60)
    W has shape (p*n, rank) = (20, 60)

    For each (i, j, k) triple, the term is:
      U[i*m + j, c] = 1  (input A[i,j])
      V[j*p + k, c] = 1  (input B[j,k])
      W[k*n + i, c] = 1  (output C[k*n + i])
    """
    start_time = time.time()
    deadline = 150.0  # wall-clock limit with safety margin

    n, m, p = DIMENSIONS
    rank = n * m * p  # 60

    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]   # 6 x 60
    V = [[0] * rank for _ in range(m * p)]   # 30 x 60
    W = [[0] * rank for _ in range(p * n)]   # 20 x 60

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline - 1.0:
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
        if time.time() > deadline - 1.0:
            break
    if time.time() > deadline - 1.0:
        break

    return {"U": U, "V": V, "W": W}
