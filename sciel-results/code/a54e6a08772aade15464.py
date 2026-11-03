# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # The tensor decomposition is sum_{c} u_c \otimes v_c \otimes w_c
    # For standard matrix multiplication, u_c, v_c, w_c are basis vectors
    # corresponding to (i,j), (j,k), (k,i) respectively.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
