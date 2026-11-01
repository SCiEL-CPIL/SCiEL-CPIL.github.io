"""Vectorized decomposition of 2x2 by 2x3 matrix multiplication tensor."""
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total = n * m * p
    col_indices = np.arange(total)
    i_flat = col_indices // (m * p)
    jk_flat = col_indices % (m * p)
    j_flat = jk_flat // p
    k_flat = jk_flat % p
    a_row = i_flat * m + j_flat
    b_row = j_flat * p + k_flat
    c_row = k_flat * n + i_flat
    U = np.zeros((n * m, total))
    V = np.zeros((m * p, total))
    W = np.zeros((p * n, total))
    U[a_row, col_indices] = 1.0
    V[b_row, col_indices] = 1.0
    W[c_row, col_indices] = 1.0
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
