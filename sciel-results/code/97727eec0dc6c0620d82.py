"""Matrix multiplication 4x4x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition (rank = n*m*p)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    for c in range(rank_schoolbook):
        i = c // (m * p)
        rem = c % (m * p)
        j = rem // p
        k = rem % p
        U_school[i * m + j, c] = 1.0
        V_school[j * p + k, c] = 1.0
        W_school[k * n + i, c] = 1.0

    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
