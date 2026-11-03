"""Matrix multiplication 2x5x5 rank-minimal exact tensor decomposition."""

import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Use the standard trivial decomposition (one rank-1 term per tensor coordinate)
    # This guarantees exactness and is the baseline.
    rank = nm * mp * pn // (m * m)  # = n * p * m = 2 * 5 * 5 = 50
    # Actually let's just use the standard loop which gives rank = n*m*p = 50
    rank = n * m * p  # 50

    U = np.zeros((nm, rank), dtype=object)
    V = np.zeros((mp, rank), dtype=object)
    W = np.zeros((pn, rank), dtype=object)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Convert to lists of lists with integer entries
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    # Ensure all entries are ints (not np.int64)
    U_list = [[int(x) for x in row] for row in U_list]
    V_list = [[int(x) for x in row] for row in V_list]
    W_list = [[int(x) for x in row] for row in W_list]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
