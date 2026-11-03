import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 105

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    i_idx, j_idx, k_idx = np.indices((n, m, p)).reshape(3, -1)
    i_flat = i_idx.ravel()
    j_flat = j_idx.ravel()
    k_flat = k_idx.ravel()

    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    cols = np.arange(rank)

    U[u_rows, cols] = 1.0
    V[v_rows, cols] = 1.0
    W[w_rows, cols] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
