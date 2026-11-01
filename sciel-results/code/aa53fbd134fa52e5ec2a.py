"""Matrix multiplication (2,2,15) tensor decomposition via schoolbook construction."""
import numpy as np

DIMENSIONS = [2, 2, 15]


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_terms = n * m * p

    U_mat = np.zeros((n * m, total_terms), dtype=float)
    V_mat = np.zeros((m * p, total_terms), dtype=float)
    W_mat = np.zeros((p * n, total_terms), dtype=float)

    for c in range(total_terms):
        i = c % n
        j = (c // n) % m
        k = c // (n * m)

        U_mat[i * m + j, c] = 1.0
        V_mat[j * p + k, c] = 1.0
        W_mat[k * n + i, c] = 1.0

    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
