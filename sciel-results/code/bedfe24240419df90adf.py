import numpy as np

def construct():
    n, m, p = 3, 4, 5
    rank = n * m * p

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strategy: iterate over output matrix indices (r, k) for the 3x5 result
    # and the inner dimension j, mapping each term to a tensor mode index
    # to ensure the W factor follows the k*n+i convention
    r = 0
    for r_idx in range(n):  # r_idx corresponds to i (row of output, 0..n-1)
        for k in range(p):  # k corresponds to column of output, 0..p-1
            for j in range(m):  # j is the inner dimension, 0..m-1
                # U factor: input matrix A is n x m, indexed as [i*m + j, r]
                # where i = r_idx
                U[r_idx * m + j][r] = 1

                # V factor: input matrix B is m x p, indexed as [j*p + k, r]
                V[j * p + k][r] = 1

                # W factor: output tensor indexed as [k*n + i, r] per convention
                W[k * n + r_idx][r] = 1

                r += 1

    return {"U": U, "V": V, "W": W}
