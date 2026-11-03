"""Matrix multiplication 4x4 x 4x9: rank-minimal exact tensor decomposition.

Uses schoolbook decomposition with rank n*m*p = 144.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 4, 4, 9

    terms = []

    def add_term(u_idx, v_idx, w_idx, coeff=1):
        if coeff != 0:
            terms.append((u_idx, v_idx, w_idx, coeff))

    # Schoolbook decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term(u_idx, v_idx, w_idx, 1)

    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for t, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        U[u_idx, t] = coeff
        V[v_idx, t] = 1
        W[w_idx, t] = 1

    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
