"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import numpy as np

def _build_factor_matrices(n, m, p):
    """Construct U, V, W factor matrices for the tensor decomposition of M_{n,m,p}."""
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    rank = n * m * p

    U = np.zeros((U_rows, rank), dtype=float)
    V = np.zeros((V_rows, rank), dtype=float)
    W = np.zeros((W_rows, rank), dtype=float)

    for r in range(rank):
        i = r // (m * p)
        j = (r // p) % m
        k = r % p
        U[i * m + j, r] = 1.0
        V[j * p + k, r] = 1.0
        W[k * n + i, r] = 1.0

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_factor_matrices(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
