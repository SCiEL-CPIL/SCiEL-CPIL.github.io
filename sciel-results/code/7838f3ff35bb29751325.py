import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Build the rank-(n*m*p) bilinear decomposition of the (n x m) x (m x p)
    # matrix-multiplication tensor using an explicit unit-tensor expansion.
    # The output factor follows the official trace convention W[k*n + i, r].
    n, m, p = DIMENSIONS
    rank = n * m * p

    # Allocate factor matrices as dense integer arrays, then emit lists.
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Each rank-1 term isolates a single (i, j, k) unit of the tensor:
    #   U picks out A[i, j],  V picks out B[j, k],  W routes to C[i, k].
    term = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, term] = 1      # A index (i, j)
                V[j * p + k, term] = 1      # B index (j, k)
                W[k * n + i, term] = 1      # C index (k, i) per trace convention
                term += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
