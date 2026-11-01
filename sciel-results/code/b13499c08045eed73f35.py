"""Matrix multiplication 2x2 by 2x13: exact tensor decomposition."""

import numpy as np

def construct():
    """Construct factor matrices for 2x2 * 2x13 matrix multiplication."""
    n, m, p = 2, 2, 13
    rank = n * m * p  # 52

    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1

    # Convert numpy arrays of dtype object to nested lists of int
    U_list = [[int(x) for x in row] for row in U.tolist()]
    V_list = [[int(x) for x in row] for row in V.tolist()]
    W_list = [[int(x) for x in row] for row in W.tolist()]

    return {"U": U_list, "V": V_list, "W": W_list}
