import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4x4 matrix multiplication via Kronecker
    product of Strassen's rank-7 decomposition for 2x2x2.
    
    The 49 terms are indexed by (c1, c2) in {0,...,6}^2.
    """
    deadline = time.time() + 120
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1]
    ])
    
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i * 4 + j, c] = U2[i1 * 2 + j1, c1] * U2[i2 * 2 + j2, c2]
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    V[j * 4 + k, c] = V2[j1 * 2 + k1, c1] * V2[j2 * 2 + k2, c2]
                                    W[k * 4 + i, c] = W2[k1 * 2 + i1, c1] * W2[k2 * 2 + i2, c2]
    
    # Convert to list of lists for the required interface
    U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n * m)]
    V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m * p)]
    W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
